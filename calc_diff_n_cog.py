"""want to calcualte the diff befween:
*  cur inf to base
*  fut inf to base
"""
from ecoshard import geoprocessing
from ecoshard import taskgraph
from osgeo import gdal
import glob
import logging
import os
import sys
import time
import numpy


logging.basicConfig(
    level=logging.DEBUG,
    stream=sys.stdout,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'))
LOGGER = logging.getLogger(os.path.splitext(os.path.basename(__file__))[0])
LOGGER.setLevel(logging.DEBUG)


def sub_rasters(raster_a, raster_b, target_raster):
    nodata_list = [
        geoprocessing.get_raster_info(path)['nodata'][0]
        for path in [raster_a, raster_b]]
    target_nodata = -9999

    def _sub_op(array_a, array_b):
        result = numpy.empty(array_a.shape)
        valid_mask = numpy.ones(result.shape, dtype=bool)
        for nd, array in zip(nodata_list, [array_a, array_b]):
            if nd is not None:
                valid_mask &= array != nd
        result[valid_mask] = array_a[valid_mask]-array_b[valid_mask]
        result[~valid_mask] = target_nodata

    geoprocessing.raster_calculator(
        [(p, 1) for p in [raster_a, raster_b]], _sub_op,
        target_raster,
        gdal.GDT_Float32,
        target_nodata)


def _make_logger_callback(message, timeout=5.0):
    """Build a timed logger callback that prints ``message`` replaced.

    Args:
        message (string): a string that expects 2 placement %% variables,
            first for % complete from ``df_complete``, second from
            ``p_progress_arg[0]``.
        timeout (float): number of seconds to wait until print

    Returns:
        Function with signature:
            logger_callback(df_complete, psz_message, p_progress_arg)

    """
    def logger_callback(df_complete, _, p_progress_arg):
        """Argument names come from the GDAL API for callbacks."""
        current_time = time.time()
        if ((current_time - logger_callback.last_time) > timeout or
                (df_complete == 1.0 and
                 logger_callback.total_time >= timeout)):
            # In some multiprocess applications I was encountering a
            # ``p_progress_arg`` of None. This is unexpected and I suspect
            # was an issue for some kind of GDAL race condition. So I'm
            # guarding against it here and reporting an appropriate log
            # if it occurs.
            progress_arg = ''
            if p_progress_arg is not None:
                progress_arg = p_progress_arg[0]

            LOGGER.info(message, df_complete * 100, progress_arg)
            logger_callback.last_time = current_time
            logger_callback.total_time += current_time
    logger_callback.last_time = time.time()
    logger_callback.total_time = 0.0

    return logger_callback


def cogit(file_path):
    # create copy with COG
    cog_driver = gdal.GetDriverByName('COG')
    base_raster = gdal.OpenEx(file_path, gdal.OF_RASTER)
    cog_dir = os.path.join(os.path.dirname(file_path), 'cog_dir')
    os.makedirs(cog_dir, exist_ok=True)
    cog_file_path = os.path.join(
        cog_dir, f'cog_{os.path.basename(file_path)}')
    options = ('COMPRESS=LZW', 'NUM_THREADS=ALL_CPUS', 'BIGTIFF=YES')
    LOGGER.info(f'convert {file_path} to COG {cog_file_path} with {options}')
    cog_raster = cog_driver.CreateCopy(
        cog_file_path, base_raster, options=options,
        callback=_make_logger_callback(
            f"COGing {cog_file_path} %.1f%% complete %s"))
    del cog_raster


def main():
    mask_dir = 'masked_for_diff'
    task_graph = taskgraph.TaskGraph(mask_dir, os.cpu_count(), 15)

    scenario_pair_list = [
        ('diff_ph_current_baseline_rcp85_2050_10', ('wwf_ph_infra_current_impact_2050_rcp85_10', 'wwf_ph_baseline_2050_rcp85_10')),
        ('diff_ph_current_baseline_rcp85_2050_90', ('wwf_ph_infra_current_impact_2050_rcp85_90', 'wwf_ph_baseline_2050_rcp85_90')),
        ('diff_ph_future_baseline_rcp85_2050_10', ('wwf_ph_infra_future_2050_rcp85_10', 'wwf_ph_baseline_2050_rcp85_10')),
        ('diff_ph_future_baseline_rcp85_2050_90', ('wwf_ph_infra_future_2050_rcp85_90', 'wwf_ph_baseline_2050_rcp85_90')),
        ]

    previously_calculated_diff = {
        ('B_sum', 'diff_ph_current_baseline_rcp85_2050_10'): r"D:\repositories\wwf-sipa\cogs_PH\upload\cog_diff_current_infra_baseline_Bsum_wwf_PH_md5_a70032.tif",
        ('B_sum', 'diff_ph_current_baseline_rcp85_2050_90'): r"D:\repositories\wwf-sipa\cogs_PH\upload\cog_diff_current_infra_baseline_Bsum_wwf_PH_md5_a70032.tif",
        ('B_sum', 'diff_ph_future_baseline_rcp85_2050_10'): r"D:\repositories\wwf-sipa\cogs_PH\upload\cog_diff_future_infra_baseline_Bsum_wwf_PH_md5_630b42.tif",
        ('B_sum', 'diff_ph_future_baseline_rcp85_2050_90'): r"D:\repositories\wwf-sipa\cogs_PH\upload\cog_diff_future_infra_baseline_Bsum_wwf_PH_md5_630b42.tif",
        ('QF', 'diff_ph_current_baseline_rcp85_2050_10'): r"D:\repositories\wwf-sipa\cogs_PH\upload\cog_diff_current_infra_baseline_QF_wwf_PH_md5_188dd7.tif",
        ('QF', 'diff_ph_current_baseline_rcp85_2050_90'): r"D:\repositories\wwf-sipa\cogs_PH\upload\cog_diff_current_infra_baseline_QF_wwf_PH_md5_188dd7.tif",
        ('QF', 'diff_ph_future_baseline_rcp85_2050_10'): r"D:\repositories\wwf-sipa\cogs_PH\upload\cog_diff_future_infra_baseline_QF_wwf_PH_md5_b9cf45.tif",
        ('QF', 'diff_ph_future_baseline_rcp85_2050_90'): r"D:\repositories\wwf-sipa\cogs_PH\upload\cog_diff_future_infra_baseline_QF_wwf_PH_md5_b9cf45.tif",
    }
    print(scenario_pair_list)
    os.makedirs(mask_dir, exist_ok=True)
    for scenario_id, scenario_pair in scenario_pair_list:
        for raster_prefix in ['QF', 'B_sum']:
            path_list = []
            mask_task_list = []
            for scenario in scenario_pair:
                base_raster_path = os.path.join(
                    f'workspace_swy_{scenario}',
                    f'{raster_prefix}_{scenario}.tif')
                if not os.path.exists(base_raster_path):
                    raise ValueError(f'{base_raster_path} not exist')

                raster_info = geoprocessing.get_raster_info(base_raster_path)
                masked_raster_path = os.path.join(
                    mask_dir, os.path.basename(base_raster_path))
                print(f'warping {base_raster_path} to {masked_raster_path}')
                mask_task = task_graph.add_task(
                    func=geoprocessing.warp_raster,
                    args=(
                        base_raster_path, raster_info['pixel_size'],
                        masked_raster_path,
                        'near'),
                    kwargs={
                        'vector_mask_options': {
                            'mask_vector_path': r"D:\repositories\wwf-sipa\data\admin_boundaries\PH_outline.gpkg"},
                        'working_dir': mask_dir},
                    target_path_list=[masked_raster_path],
                    task_name=f'mask {masked_raster_path}')
                _ = task_graph.add_task(
                    func=cogit,
                    args=(masked_raster_path,),
                    dependent_task_list=[mask_task],
                    task_name=f'cog {masked_raster_path}')
                path_list.append(masked_raster_path)
                mask_task_list.append(mask_task)

            diff_target_path = os.path.join(
                mask_dir, f'{scenario_id}_{raster_prefix}.tif')
            sub_task = task_graph.add_task(
                func=sub_rasters,
                args=(path_list[0], path_list[1], diff_target_path),
                dependent_task_list=mask_task_list,
                target_path_list=[diff_target_path],
                task_name=f'diff {raster_prefix} {scenario_id}')

            _ = task_graph.add_task(
                func=cogit,
                args=(diff_target_path,),
                dependent_task_list=[sub_task],
                task_name=f'cog {diff_target_path}')

            second_order_target_path = os.path.join(
                mask_dir, f'second_order_{scenario_id}_from_2020_{raster_prefix}.tif')
            second_order_sub_task = task_graph.add_task(
                func=sub_rasters,
                args=(
                    diff_target_path,
                    previously_calculated_diff[(raster_prefix, scenario_id)],
                    second_order_target_path),
                dependent_task_list=[sub_task],
                target_path_list=[second_order_target_path],
                task_name=f'diff {raster_prefix} {scenario_id}')

            _ = task_graph.add_task(
                func=cogit,
                args=(second_order_target_path,),
                dependent_task_list=[second_order_sub_task],
                task_name=f'cog {second_order_target_path}')

    task_graph.join()
    task_graph.close()
    LOGGER.info('all done!')


if __name__ == '__main__':
    main()
