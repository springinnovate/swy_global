"""Run SWY globally."""
import re
from datetime import datetime
import argparse
import collections
import configparser
import glob
import logging
import multiprocessing
import os
import shutil
import threading
import time

from inspring.seasonal_water_yield import seasonal_water_yield
from ecoshard import geoprocessing
from ecoshard import taskgraph
from osgeo import gdal
from osgeo import ogr
from osgeo import osr

gdal.UseExceptions()

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
logging.getLogger('ecoshard.taskgraph').setLevel(logging.INFO)
logging.getLogger('ecoshard.ecoshard').setLevel(logging.INFO)
logging.getLogger('urllib3.connectionpool').setLevel(logging.INFO)
logging.getLogger('ecoshard.geoprocessing.geoprocessing').setLevel(
    logging.ERROR)
logging.getLogger('ecoshard.geoprocessing.routing.routing').setLevel(
    logging.WARNING)
logging.getLogger('ecoshard.geoprocessing.geoprocessing_core').setLevel(
    logging.ERROR)
logging.getLogger('inspring.seasonal_water_yield').setLevel(logging.WARNING)

LOGGER = logging.getLogger(__name__)

GLOBAL_INI_PATH = 'swy_global.ini'

N_TO_BUFFER_STITCH = 10


def _clean_workspace_worker(
        expected_signal_count, stitch_done_queue, keep_intermediate_files):
    """Removes workspaces when completed.

    Args:
        expected_signal_count (int): the number of times to be notified
            of a done path before it should be deleted.
        stitch_done_queue (queue): will contain directory paths with the
            same directory path appearing `expected_signal_count` times,
            the directory will be removed. Recieving `None` will terminate
            the process.
        keep_intermediate_files (bool): keep intermediate files if true

    Returns:
        None
    """
    try:
        count_dict = collections.defaultdict(int)
        while True:
            dir_path = stitch_done_queue.get()
            if dir_path is None:
                LOGGER.info('recieved None, quitting clean_workspace_worker')
                return
            count_dict[dir_path] += 1
            if count_dict[dir_path] == expected_signal_count:
                LOGGER.info(
                    f'removing {dir_path} after {count_dict[dir_path]} '
                    f'signals')
                if not keep_intermediate_files:
                    shutil.rmtree(dir_path)
                del count_dict[dir_path]
    except Exception:
        LOGGER.exception('error on clean_workspace_worker')


def _process_scenario_ini(scenario_config_path):
    """Verify that the ini file has correct structure.

    Args:
        ini_config (configparser obj): config parsed object with
            'expected_keys' and '`scenario_id`' as fields.

    Returns
        (configparser object, scenario id)

    Raises errors if config file not formatted correctly
    """
    global_config = configparser.ConfigParser(allow_no_value=True)
    global_config.read(GLOBAL_INI_PATH)
    global_config.read(scenario_config_path)
    scenario_id = os.path.basename(os.path.splitext(scenario_config_path)[0])
    if scenario_id not in global_config:
        raise ValueError(
            f'expected a section called [{scenario_id}] in configuration file'
            f'but was not found')
    scenario_config = global_config[scenario_id]
    missing_keys = []
    for key in global_config['expected_keys']:
        if key not in scenario_config:
            missing_keys.append(key)
    if missing_keys:
        raise ValueError(
            f'expected the following keys in "{scenario_config_path}" '
            f'but not found: "{", ".join(missing_keys)}"')
    LOGGER.debug(scenario_config)
    for key in scenario_config:
        if key.endswith('_path'):
            possible_path = scenario_config[key]
            if not os.path.exists(possible_path):
                raise ValueError(
                    f'expected a file from "{key}" at "{possible_path}" '
                    f'but file not found')

    return scenario_config, scenario_id


def clip_raster_by_vector(
        raster_path, vector_path, vector_field, field_value_list,
        target_raster_path, target_vector_path):
    """Clip and mask raster to vector using tightest bounding box."""
    raster_info = geoprocessing.get_raster_info(raster_path)
    raster_projection_wkt = raster_info['projection_wkt']

    LOGGER.info(f'reproject vector to {target_vector_path}')
    where_filter = f'{vector_field.upper()} IN (' + ', '.join([str(x) for x in field_value_list]) + ')'
    LOGGER.debug(f'{target_vector_path} {where_filter}')
    geoprocessing.reproject_vector(
        vector_path, raster_projection_wkt, target_vector_path,
        where_filter=where_filter, driver_name='GPKG')

    projected_vector_info = geoprocessing.get_vector_info(
        target_vector_path)

    target_bb = geoprocessing.merge_bounding_box_list(
        [raster_info['bounding_box'], projected_vector_info['bounding_box']],
        'intersection')

    geoprocessing.warp_raster(
        raster_path, raster_info['pixel_size'], target_raster_path,
        'near', target_bb=target_bb,
        vector_mask_options={'mask_vector_path': target_vector_path},
        working_dir=os.getcwd())
    LOGGER.info(f'all done, raster at {target_raster_path}')


def _create_fid_subset(
        base_vector_path, fid_list, target_epsg, target_vector_path):
    """Create subset of vector that matches fid list, projected into epsg."""
    vector = gdal.OpenEx(base_vector_path, gdal.OF_VECTOR)
    layer = vector.GetLayer()
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(target_epsg)
    layer.SetAttributeFilter(
        f'"FID" in ('
        f'{", ".join([str(v) for v in fid_list])})')
    feature_count = layer.GetFeatureCount()
    gpkg_driver = ogr.GetDriverByName('gpkg')
    unprojected_vector_path = '%s_wgs84%s' % os.path.splitext(
        target_vector_path)
    subset_vector = gpkg_driver.CreateDataSource(unprojected_vector_path)
    subset_vector.CopyLayer(
        layer, os.path.basename(os.path.splitext(target_vector_path)[0]))
    geoprocessing.reproject_vector(
        unprojected_vector_path, srs.ExportToWkt(), target_vector_path,
        driver_name='gpkg', copy_fields=False)
    subset_vector = None
    layer = None
    vector = None
    gpkg_driver.DeleteDataSource(unprojected_vector_path)
    target_vector = gdal.OpenEx(target_vector_path, gdal.OF_VECTOR)
    target_layer = target_vector.GetLayer()
    if feature_count != target_layer.GetFeatureCount():
        raise ValueError(
            f'expected {feature_count} in {target_vector_path} but got '
            f'{target_layer.GetFeatureCount()}')


def _batch_into_watershed_subsets(
        watershed_root_dir, degree_separation, done_token_path,
        global_bb, target_subset_dir, watershed_subset=None):
    """Construct geospatially adjacent subsets.

    Breaks watersheds up into geospatially similar watersheds and limits
    the upper size to no more than specified area. This allows for a
    computationally efficient batch to run on a large contiguous area in
    parallel while avoiding batching watersheds that are too small.

    Args:
        watershed_root_dir (str): path to watershed .shp files.
        degree_separation (int): a blocksize number of degrees to coalasce
            watershed subsets into.
        done_token_path (str): path to file to write when function is
            complete, indicates for batching that the task is complete.
        global_bb (list): min_lng, min_lat, max_lng, max_lat bounding box
            to limit watersheds to be selected from
        target_subset_dir (str); path to directory to store watershed subsets
        watershed_subset (tuple): if not None, tuple of
            (basename, feature_key, (typle, of, values)).

    Returns:
        list of (job_id, watershed.gpkg) tuples where the job_id is a
        unique identifier for that subwatershed set and watershed.gpkg is
        a subset of the original global watershed files.

    """
    # ensures we don't have more than 1000 watersheds per job
    task_graph = taskgraph.TaskGraph(
        watershed_root_dir, multiprocessing.cpu_count(), 10,
        taskgraph_name='batch watersheds')
    watershed_path_area_list = []
    job_id_set = set()
    LOGGER.debug(f'looping through all the shp files in {watershed_root_dir}')
    for watershed_path in glob.glob(
            os.path.join(watershed_root_dir, '*.shp')):
        LOGGER.debug(f'scheduling {os.path.basename(watershed_path)} *********')
        subbatch_job_index_map = collections.defaultdict(int)
        # lambda describes the FIDs to process per job, the list of lat/lng
        # bounding boxes for each FID, and the total degree area of the job
        watershed_fid_index = collections.defaultdict(
            lambda: [list(), list(), 0])
        watershed_basename = os.path.splitext(
            os.path.basename(watershed_path))[0]
        watershed_ids = None
        watershed_vector = gdal.OpenEx(watershed_path, gdal.OF_VECTOR)
        watershed_layer = watershed_vector.GetLayer()

        if watershed_subset:
            LOGGER.debug(f'testing against {watershed_subset[0]}')
            if watershed_basename != watershed_subset[0]:
                LOGGER.debug(f'skipping because {watershed_basename} != {watershed_subset[0]}')
                continue
            else:
                # just grab the subset
                watershed_ids = watershed_subset[2]
                watershed_layer = [
                    watershed_layer.GetFeature(fid) for fid in watershed_ids]
                LOGGER.debug(f'getting that subset of {watershed_ids}')

        # watershed layer is either the layer or a list of features
        for watershed_feature in watershed_layer:
            fid = watershed_feature.GetFID()
            watershed_geom = watershed_feature.GetGeometryRef()
            watershed_centroid = watershed_geom.Centroid()
            epsg = geoprocessing.get_utm_zone(
                watershed_centroid.GetX(), watershed_centroid.GetY())
            if watershed_geom.Area() > 1 or watershed_ids:
                # one degree grids or immediates get special treatment
                job_id = (f'{watershed_basename}_{fid}', epsg)
                watershed_fid_index[job_id][0] = [fid]
            else:
                # clamp into degree_separation squares
                x, y = [
                    int(v//degree_separation)*degree_separation for v in (
                        watershed_centroid.GetX(), watershed_centroid.GetY())]
                base_job_id = f'{watershed_basename}_{x}_{y}'
                # keep the epsg in the string because the centroid might lie
                # on a different boundary
                job_id = (f'''{base_job_id}_{
                    subbatch_job_index_map[base_job_id]}_{epsg}''', epsg)
                if len(watershed_fid_index[job_id][0]) > 1000:
                    subbatch_job_index_map[base_job_id] += 1
                    job_id = (f'''{base_job_id}_{
                        subbatch_job_index_map[base_job_id]}_{epsg}''', epsg)
                watershed_fid_index[job_id][0].append(fid)
            watershed_envelope = watershed_geom.GetEnvelope()
            watershed_bb = [watershed_envelope[i] for i in [0, 2, 1, 3]]
            if (watershed_bb[0] < global_bb[0] or
                    watershed_bb[2] > global_bb[2] or
                    watershed_bb[1] > global_bb[3] or
                    watershed_bb[3] < global_bb[1]):
                # LOGGER.warn(
                #     f'{watershed_bb} is on a dangerous boundary so dropping')
                # drop because it's outside of the BB
                watershed_fid_index[job_id][0].pop()
                continue
            watershed_fid_index[job_id][1].append(watershed_bb)
            watershed_fid_index[job_id][2] += watershed_geom.Area()

        watershed_geom = None
        watershed_feature = None

        for (job_id, epsg), (fid_list, watershed_envelope_list, area) in \
                sorted(
                    watershed_fid_index.items(), key=lambda x: x[1][-1],
                    reverse=True):
            if job_id in job_id_set:
                raise ValueError(f'{job_id} already processed')
            if len(watershed_envelope_list) < 3 and area < 1e-6:
                # it's too small to process
                #LOGGER.debug(f'TOPOO AMSKK TO PROCESS {watershed_envelope_list} {area}')
                continue
            job_id_set.add(job_id)

            watershed_subset_path = os.path.join(
                target_subset_dir, f'{job_id}_a{area:.3f}.gpkg')
            if not os.path.exists(watershed_subset_path):
                task_graph.add_task(
                    func=_create_fid_subset,
                    args=(
                        watershed_path, fid_list, epsg, watershed_subset_path),
                    target_path_list=[watershed_subset_path],
                    task_name=job_id)
            watershed_path_area_list.append(
                (area, watershed_subset_path))
            LOGGER.debug(f'added {watershed_subset_path}****')

        watershed_layer = None
        watershed_vector = None

    task_graph.join()
    task_graph.close()
    task_graph = None

    # create a global sorted watershed path list so it's sorted by area overall
    # not just by region per area
    with open(done_token_path, 'w') as token_file:
        token_file.write(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    sorted_watershed_path_list = [
        path for area, path in sorted(watershed_path_area_list, reverse=True)]
    return sorted_watershed_path_list


def _calculate_intersecting_bounding_box(raster_path_list):
    # create intersecting bounding box of input data
    raster_info_list = [
        geoprocessing.get_raster_info(raster_path)
        for raster_path in raster_path_list
        if geoprocessing.get_raster_info(raster_path)['projection_wkt']
        is not None]

    raster_bounding_box_list = [
        geoprocessing.transform_bounding_box(
            info['bounding_box'],
            info['projection_wkt'],
            osr.SRS_WKT_WGS84_LAT_LONG)
        for info in raster_info_list]

    target_bounding_box = geoprocessing.merge_bounding_box_list(
        raster_bounding_box_list, 'intersection')
    LOGGER.info(f'calculated target_bounding_box: {target_bounding_box}')
    return target_bounding_box


def stitch_worker(
        rasters_to_stitch_queue, target_stitch_raster_path, n_expected,
        signal_done_queue):
    """Update the database with completed work.

    Args:
        rasters_to_stitch_queue (queue): queue that recieves paths to
            rasters to stitch into target_stitch_raster_path.
        target_stitch_raster_path (str): path to an existing raster to stitch
            into.
        n_expected (int): number of expected stitch signals
        signal_done_queue (queue): as each job is complete the directory path
            to the raster will be passed in to eventually remove.


    Return:
        ``None``
    """
    try:
        processed_so_far = 0
        n_buffered = 0
        start_time = time.time()
        stitch_buffer_list = []
        LOGGER.info(f'started stitch worker for {target_stitch_raster_path}')
        while True:
            payload = rasters_to_stitch_queue.get()
            if payload is not None:
                if payload[0] is None:  # means skip this raster
                    processed_so_far += 1
                    continue
                stitch_buffer_list.append(payload)

            if len(stitch_buffer_list) > N_TO_BUFFER_STITCH or payload is None:
                LOGGER.info(
                    f'about to stitch {n_buffered} into '
                    f'{target_stitch_raster_path}')
                geoprocessing.stitch_rasters(
                    stitch_buffer_list, ['near']*len(stitch_buffer_list),
                    (target_stitch_raster_path, 1),
                    area_weight_m2_to_wgs84=True,
                    overlap_algorithm='replace')
                #  _ is the band number
                for stitch_path, _ in stitch_buffer_list:
                    signal_done_queue.put(os.path.dirname(stitch_path))
                stitch_buffer_list = []

            if payload is None:
                LOGGER.info(f'all done sitching {target_stitch_raster_path}')
                return

            processed_so_far += 1
            jobs_per_sec = processed_so_far / (time.time() - start_time)
            remaining_time_s = (
                n_expected / jobs_per_sec)
            remaining_time_h = int(remaining_time_s // 3600)
            remaining_time_s -= remaining_time_h * 3600
            remaining_time_m = int(remaining_time_s // 60)
            remaining_time_s -= remaining_time_m * 60
            LOGGER.info(
                f'remaining jobs to process for {target_stitch_raster_path}: '
                f'{n_expected-processed_so_far} - '
                f'processed so far {processed_so_far} - '
                f'process/sec: {jobs_per_sec:.1f}s - '
                f'time left: {remaining_time_h}:'
                f'{remaining_time_m:02d}:{remaining_time_s:04.1f}')
    except Exception:
        LOGGER.exception(
            f'error on stitch worker for {target_stitch_raster_path}')
        raise


def _run_swy(
        task_graph,
        model_args,
        target_stitch_raster_map,
        global_pixel_size_deg,
        target_pixel_size,
        keep_intermediate_files,
        watershed_path_list,
        result_suffix):
    """Run SWY

    This function will iterate through the watershed subset list, run the SDR
    model on those subwatershed regions, and stitch those data back into a
    global raster.

    Args:
        task_graph (TaskGraph): to schedule parallalization and avoid
            recompuation
        model_args (dict): model args to pass to SWY model.
        target_stitch_raster_map (dict): maps the local path of an output
            raster of this model to an existing global raster to stich into.
        keep_intermediate_files (bool): if True, the intermediate watershed
            workspace created underneath `workspace_dir` is deleted.
        result_suffix (str): optional, prepended to the global stitch results.

    Returns:
        None.
    """
    # create intersecting bounding box of input data, this ignores the
    # precip and eto rasters, but I think is okay so we can avoid parsing
    # those out here and the lulc will likely drive the AOI
    model_args = model_args.copy()
    global_wgs84_bb = _calculate_intersecting_bounding_box(
        [model_args[key] for key in [
            'dem_raster_path', 'lulc_raster_path',
            'soil_group_path']])

    # create global stitch rasters and start workers
    stitch_raster_queue_map = {}
    stitch_worker_list = []
    multiprocessing_manager = multiprocessing.Manager()
    signal_done_queue = multiprocessing_manager.Queue()
    for local_result_path, global_stitch_raster_path in \
            target_stitch_raster_map.items():
        if result_suffix is not None:
            global_stitch_raster_path = (
                f'%s_{result_suffix}%s' % os.path.splitext(
                    global_stitch_raster_path))
            local_result_path = (
                f'%s_{result_suffix}%s' % os.path.splitext(
                    local_result_path))
        if not os.path.exists(global_stitch_raster_path):
            LOGGER.info(f'creating {global_stitch_raster_path}')
            driver = gdal.GetDriverByName('GTiff')
            n_cols = int((global_wgs84_bb[2]-global_wgs84_bb[0])/global_pixel_size_deg)
            n_rows = int((global_wgs84_bb[3]-global_wgs84_bb[1])/global_pixel_size_deg)
            LOGGER.info(f'**** creating raster of size {n_cols} by {n_rows}')
            target_raster = driver.Create(
                global_stitch_raster_path,
                n_cols, n_rows, 1,
                gdal.GDT_Float32,
                options=(
                    'TILED=YES', 'BIGTIFF=YES', 'COMPRESS=LZW', 'PREDICTOR=2',
                    'SPARSE_OK=TRUE', 'BLOCKXSIZE=256', 'BLOCKYSIZE=256'))
            wgs84_srs = osr.SpatialReference()
            wgs84_srs.ImportFromEPSG(4326)
            target_raster.SetProjection(wgs84_srs.ExportToWkt())
            target_raster.SetGeoTransform(
                [global_wgs84_bb[0], global_pixel_size_deg, 0,
                 global_wgs84_bb[3], 0, -global_pixel_size_deg])
            target_band = target_raster.GetRasterBand(1)
            target_band.SetNoDataValue(-9999)
            target_raster = None
        stitch_queue = multiprocessing_manager.Queue(N_TO_BUFFER_STITCH*2)
        stitch_thread = threading.Thread(
            target=stitch_worker,
            args=(
                stitch_queue, global_stitch_raster_path,
                len(watershed_path_list),
                signal_done_queue))
        stitch_thread.start()
        stitch_raster_queue_map[local_result_path] = stitch_queue
        stitch_worker_list.append(stitch_thread)

    clean_workspace_worker = threading.Thread(
        target=_clean_workspace_worker,
        args=(len(target_stitch_raster_map), signal_done_queue,
              keep_intermediate_files))
    clean_workspace_worker.daemon = True
    clean_workspace_worker.start()

    # Iterate through each watershed subset and run SDR
    # stitch the results of whatever outputs to whatever global output raster.
    scheduled_watershed_set = set()
    for index, watershed_path in enumerate(watershed_path_list):
        local_workspace_dir = os.path.join(
            model_args['workspace_dir'], os.path.splitext(
                os.path.basename(watershed_path))[0])
        if local_workspace_dir in scheduled_watershed_set:
            raise ValueError(
                f'somehow {local_workspace_dir} has been added twice, here is '
                f'watershed path list {watershed_path_list}')
        task_name = f'sdr {os.path.basename(local_workspace_dir)}'
        task_graph.add_task(
            func=_execute_swy_job,
            args=(
                global_wgs84_bb, watershed_path, local_workspace_dir,
                model_args, stitch_raster_queue_map,
                target_pixel_size, result_suffix),
            transient_run=False,
            priority=-index,  # priority in insert order
            task_name=task_name)

    LOGGER.info('wait for SDR jobs to complete')
    task_graph.join()
    for local_result_path, stitch_queue in stitch_raster_queue_map.items():
        stitch_queue.put(None)
    LOGGER.info('all done with SDR, waiting for stitcher to terminate')
    for stitch_thread in stitch_worker_list:
        stitch_thread.join()
    LOGGER.info(
        'all done with stitching, waiting for workspace worker to terminate')
    signal_done_queue.put(None)
    clean_workspace_worker.join()

    LOGGER.info('all done with SDR -- stitcher terminated')


def _watersheds_intersect(wgs84_bb, watersheds_path):
    """True if watersheds intersect the wgs84 bounding box."""
    watershed_info = geoprocessing.get_vector_info(watersheds_path)
    watershed_wgs84_bb = geoprocessing.transform_bounding_box(
        watershed_info['bounding_box'],
        watershed_info['projection_wkt'],
        osr.SRS_WKT_WGS84_LAT_LONG)
    try:
        _ = geoprocessing.merge_bounding_box_list(
            [wgs84_bb, watershed_wgs84_bb], 'intersection')
        LOGGER.info(f'{watersheds_path} intersects {wgs84_bb} with {watershed_wgs84_bb}')
        return True
    except ValueError:
        LOGGER.warn(f'{watersheds_path} does not intersect {wgs84_bb}')
        return False


def _warp_raster_stack(
        base_raster_path_list, warped_raster_path_list,
        resample_method_list, clip_pixel_size, target_pixel_size,
        clip_bounding_box, clip_projection_wkt, watershed_clip_vector_path):
    """Do an align of all the rasters.

    Arguments are same as geoprocessing.align_and_resize_raster_stack.
    """
    for raster_path, warped_raster_path, resample_method in zip(
            base_raster_path_list, warped_raster_path_list,
            resample_method_list):
        LOGGER.debug(f'warp {raster_path} to {warped_raster_path}')
        _clip_and_warp(
            raster_path, clip_bounding_box, clip_pixel_size, resample_method,
            clip_projection_wkt, watershed_clip_vector_path, target_pixel_size,
            warped_raster_path)


def _clip_and_warp(
        base_raster_path, clip_bounding_box, clip_pixel_size, resample_method,
        clip_projection_wkt, watershed_clip_vector_path, target_pixel_size,
        warped_raster_path):
    working_dir = os.path.dirname(warped_raster_path)
    # first clip to clip projection
    clipped_raster_path = '%s_clipped%s' % os.path.splitext(
        warped_raster_path)
    geoprocessing.warp_raster(
        base_raster_path, clip_pixel_size, clipped_raster_path,
        resample_method, **{
            'target_bb': clip_bounding_box,
            'target_projection_wkt': clip_projection_wkt,
            'working_dir': working_dir
        })

    # second, warp and mask to vector
    watershed_projection_wkt = geoprocessing.get_vector_info(
        watershed_clip_vector_path)['projection_wkt']
    vector_mask_options = {'mask_vector_path': watershed_clip_vector_path}
    geoprocessing.warp_raster(
        clipped_raster_path, (target_pixel_size, -target_pixel_size),
        warped_raster_path, resample_method, **{
            'target_projection_wkt': watershed_projection_wkt,
            'vector_mask_options': vector_mask_options,
            'working_dir': working_dir,
        })
    os.remove(clipped_raster_path)


def _execute_swy_job(
        global_wgs84_bb, watersheds_path, local_workspace_dir, model_args,
        stitch_raster_queue_map, target_pixel_size, result_suffix):
    """Worker to execute sdr and send signals to stitcher.

    Args:
        global_wgs84_bb (list): bounding box to limit run to, if watersheds do
            not fit, then skip
        watersheds_path (str): path to watershed to run model over
        local_workspace_dir (str): path to local directory
        model_args (dict): for running model.
        target_pixel_size (float): target pixel size


        stitch_raster_queue_map (dict): map of local result path to
            the stitch queue to signal when job is done.

    Returns:
        None.
    """
    if not _watersheds_intersect(global_wgs84_bb, watersheds_path):
        LOGGER.debug(f'{watersheds_path} does not overlap {global_wgs84_bb}')
        for local_result_path, stitch_queue in stitch_raster_queue_map.items():
            # indicate skipping
            stitch_queue.put((None, 1))
        return

    dem_pixel_size = geoprocessing.get_raster_info(
        model_args['dem_raster_path'])['pixel_size']

    # TODO: parse out all the ETO and precip rasters
    path_list = [model_args[key] for key in [
        'dem_raster_path', 'soil_group_path', 'lulc_raster_path']]

    clipped_data_dir = os.path.join(local_workspace_dir, 'data')
    os.makedirs(clipped_data_dir, exist_ok=True)
    warped_raster_path_list = [
        os.path.join(clipped_data_dir, os.path.basename(path))
        for path in path_list]
    resample_method_list = ['bilinear', 'mode', 'mode']
    LOGGER.debug(path_list)
    LOGGER.debug(warped_raster_path_list)

    model_args = model_args.copy()
    model_args['workspace_dir'] = local_workspace_dir
    model_args['dem_raster_path'] = warped_raster_path_list[0]
    model_args['soil_group_path'] = warped_raster_path_list[1]
    model_args['lulc_raster_path'] = warped_raster_path_list[2]

    local_et0_dir = os.path.join(clipped_data_dir, 'local_et0')
    local_precip_dir = os.path.join(clipped_data_dir, 'local_precip')
    month_based_rasters = collections.defaultdict(list)
    for month_index in range(1, 13):
        month_file_match = re.compile(r'.*[^\d]0?%d\.tif$' % month_index)
        for data_type, dir_path in [
                ('et0', model_args['et0_dir']),
                ('Precip', model_args['precip_dir'])]:
            file_list = [
                month_file_path for month_file_path in glob.glob(
                    os.path.join(dir_path, '*.tif'))
                if month_file_match.match(month_file_path)]
            if len(file_list) == 0:
                raise ValueError(
                    "No %s found for month %d" % (data_type, month_index))
            if len(file_list) > 1:
                raise ValueError(
                    "Ambiguous set of files found for month %d: %s" %
                    (month_index, file_list))
            month_based_rasters[data_type].append(file_list[0])

    base_raster_path_list = (
        path_list + month_based_rasters['et0'] + month_based_rasters['Precip'])

    warped_raster_path_list += [
        os.path.join(local_et0_dir, os.path.basename(path))
        for path in month_based_rasters['et0']]
    resample_method_list += ['bilinear']*len(month_based_rasters['et0'])
    os.makedirs(os.path.join(local_et0_dir), exist_ok=True)
    model_args['et0_dir'] = local_et0_dir

    warped_raster_path_list += [
        os.path.join(local_precip_dir, os.path.basename(path))
        for path in month_based_rasters['Precip']]
    resample_method_list += ['bilinear']*len(month_based_rasters['Precip'])
    os.makedirs(os.path.join(local_precip_dir), exist_ok=True)
    model_args['precip_dir'] = local_precip_dir

    watershed_info = geoprocessing.get_vector_info(watersheds_path)
    target_projection_wkt = watershed_info['projection_wkt']
    watershed_bb = watershed_info['bounding_box']
    lat_lng_bb = geoprocessing.transform_bounding_box(
        watershed_bb, target_projection_wkt, osr.SRS_WKT_WGS84_LAT_LONG)

    # re-warp stuff we already did
    _warp_raster_stack(
        base_raster_path_list, warped_raster_path_list,
        resample_method_list, dem_pixel_size, target_pixel_size,
        lat_lng_bb, osr.SRS_WKT_WGS84_LAT_LONG, watersheds_path)

    model_args['aoi_path'] = watersheds_path
    model_args['prealigned'] = True
    model_args['reuse_dem'] = True
    model_args['single_outlet'] = geoprocessing.get_vector_info(
        watersheds_path)['feature_count'] == 1

    if 'soil_hydrologic_map' in model_args:
        map_letter_to_int = {
            'a': 1, 'b': 2, 'c': 3, 'd': 4
        }
        reclass_map = {
            lucode: map_letter_to_int[soil_code.lower()]
            for lucode, soil_code in model_args['soil_hydrologic_map'].items()
        }

        reclass_soil_group_path = os.path.join(
            local_workspace_dir,
            f"reclassed_{os.path.basename(model_args['soil_group_path'])}")
        geoprocessing.reclassify_raster(
            (model_args['soil_group_path'], 1),
            reclass_map, reclass_soil_group_path, gdal.GDT_Byte, 0,
            values_required=True)

        model_args['soil_group_path'] = (
            reclass_soil_group_path)

    # TODO: need to add these special flags so things don't re-warp

    seasonal_water_yield.execute(model_args)
    for local_result_path, stitch_queue in stitch_raster_queue_map.items():
        stitch_queue.put(
            (os.path.join(model_args['workspace_dir'], local_result_path), 1))


def main():
    parser = argparse.ArgumentParser(description='Global SWY')
    parser.add_argument(
        'scenario_config_path',
        help='Pattern to .INI file(s) that describes scenario(s) to run.')
    args = parser.parse_args()

    scenario_config_path_list = list(glob.glob(args.scenario_config_path))
    LOGGER.info(f'''parsing and validating {
        len(scenario_config_path_list)} configuration files''')
    config_scenario_list = []
    for scenario_config_path in scenario_config_path_list:
        scenario_config, scenario_id = _process_scenario_ini(
            scenario_config_path)
        config_scenario_list.append((scenario_config, scenario_id))

    for scenario_config, scenario_id in config_scenario_list:

        LOGGER.debug(f'{scenario_config}:, {scenario_id}')

        local_workspace = f'workspace_swy_{scenario_id}'
        os.makedirs(local_workspace, exist_ok=True)
        task_graph = taskgraph.TaskGraph(
            local_workspace, multiprocessing.cpu_count(), 5.0)

        exclusive_watershed_subset = scenario_config.get(
            'watershed_subset', fallback=None)
        if exclusive_watershed_subset is not None:
            exclusive_watershed_subset = eval(exclusive_watershed_subset)
        watershed_subset_token = os.path.join(
            local_workspace, 'watershed_subset.token')
        watershed_subset_dir = os.path.join(
            local_workspace, 'watershed_subset_files')
        os.makedirs(watershed_subset_dir, exist_ok=True)

        global_wgs84_bb = _calculate_intersecting_bounding_box(
            [scenario_config[key] for key in [
                'dem_raster_path', 'lulc_raster_path',
                'soil_hydrologic_group_raster_path']])

        watershed_subset_task = task_graph.add_task(
            func=_batch_into_watershed_subsets,
            args=(
                scenario_config['watersheds_vector_path'], 4,
                watershed_subset_token,
                global_wgs84_bb,
                watershed_subset_dir,
                exclusive_watershed_subset),
            target_path_list=[watershed_subset_token],
            store_result=True,
            task_name='watershed subset batch')
        watershed_subset_list = watershed_subset_task.get()
        LOGGER.debug(watershed_subset_list)

        model_args = {
            'workspace_dir': local_workspace,
            'results_suffix': scenario_id,
            'threshold_flow_accumulation': float(scenario_config['threshold_flow_accumulation']),
            'et0_dir': scenario_config['et0_dir'],
            'precip_dir': scenario_config['precip_dir'],
            'dem_raster_path': scenario_config['dem_raster_path'],
            'lulc_raster_path': scenario_config['lulc_raster_path'],
            'soil_group_path': scenario_config['soil_hydrologic_group_raster_path'],
            'biophysical_table_path': scenario_config['biophysical_table_path'],
            'rain_events_table_path': scenario_config['rain_events_table_path'],
            'monthly_alpha': False,
            'alpha_m': 1/12,
            'beta_i': scenario_config['beta_i'],
            'gamma': scenario_config['gamma'],
            'user_defined_local_recharge': None,
            'user_defined_climate_zones': None,
            'lucode_field': scenario_config['lucode_field'],
        }

        if 'soil_hydrologic_map' in scenario_config:
            model_args['soil_hydrologic_map'] = eval(scenario_config['soil_hydrologic_map'])

        swy_target_stitch_raster_map = {
            "B.tif": os.path.join(
                model_args['workspace_dir'], 'B.tif'),
            "B_sum.tif": os.path.join(
                model_args['workspace_dir'], 'B_sum.tif'),
            "L_avail.tif": os.path.join(
                model_args['workspace_dir'], 'L_avail.tif'),
            "L_sum_avail.tif": os.path.join(
                model_args['workspace_dir'], 'L_sum_avail.tif'),
            "QF.tif": os.path.join(
                model_args['workspace_dir'], 'QF.tif'),
            os.path.join('intermediate_outputs', 'qf_1.tif'): os.path.join(model_args['workspace_dir'], 'qf_1.tif'),
            os.path.join('intermediate_outputs', 'qf_2.tif'): os.path.join(model_args['workspace_dir'], 'qf_2.tif'),
            os.path.join('intermediate_outputs', 'qf_3.tif'): os.path.join(model_args['workspace_dir'], 'qf_3.tif'),
            os.path.join('intermediate_outputs', 'qf_4.tif'): os.path.join(model_args['workspace_dir'], 'qf_4.tif'),
            os.path.join('intermediate_outputs', 'qf_5.tif'): os.path.join(model_args['workspace_dir'], 'qf_5.tif'),
            os.path.join('intermediate_outputs', 'qf_6.tif'): os.path.join(model_args['workspace_dir'], 'qf_6.tif'),
            os.path.join('intermediate_outputs', 'qf_7.tif'): os.path.join(model_args['workspace_dir'], 'qf_7.tif'),
            os.path.join('intermediate_outputs', 'qf_8.tif'): os.path.join(model_args['workspace_dir'], 'qf_8.tif'),
            os.path.join('intermediate_outputs', 'qf_9.tif'): os.path.join(model_args['workspace_dir'], 'qf_9.tif'),
            os.path.join('intermediate_outputs', 'qf_10.tif'): os.path.join(model_args['workspace_dir'], 'qf_10.tif'),
            os.path.join('intermediate_outputs', 'qf_11.tif'): os.path.join(model_args['workspace_dir'], 'qf_11.tif'),
            os.path.join('intermediate_outputs', 'qf_12.tif'): os.path.join(model_args['workspace_dir'], 'qf_12.tif'),
        }
        keep_intermediate_files = False
        _run_swy(
            task_graph=task_graph,
            model_args=model_args,
            target_stitch_raster_map=swy_target_stitch_raster_map,
            global_pixel_size_deg=float(scenario_config['GLOBAL_PIXEL_SIZE_DEG']),
            keep_intermediate_files=keep_intermediate_files,
            watershed_path_list=watershed_subset_list,
            target_pixel_size=eval(scenario_config['target_pixel_size']),
            result_suffix=scenario_id)

        task_graph.join()
        task_graph.close()

        # LOGGER.debug(local_data_path_map)
        # seasonal_water_yield.execute(local_data_path_map)


if __name__ == '__main__':
    main()
