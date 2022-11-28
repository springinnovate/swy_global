"""Run SWY globally."""
import tempfile
from datetime import datetime
import argparse
import collections
import configparser
import glob
import gzip
import itertools
import logging
import multiprocessing
import os
import shutil
import threading
import time

from inspring import seasonal_water_yield
from ecoshard import geoprocessing
from ecoshard import taskgraph
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import ecoshard
import requests

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
WORKSPACE_DIR = 'swy_workspace'
os.makedirs(WORKSPACE_DIR, exist_ok=True)


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
        target_raster_path):
    """Clip and mask raster to vector using tightest bounding box."""
    temp_dir = os.path.join(
        os.path.dirname(target_raster_path), 'clip_raster_workspace')
    os.makedirs(temp_dir, exist_ok=True)

    raster_info = geoprocessing.get_raster_info(raster_path)
    raster_projection_wkt = raster_info['projection_wkt']

    projected_vector_path = os.path.splitext(os.path.join(
        temp_dir, os.path.basename(vector_path)))[0] + '.gpkg'

    LOGGER.info(f'reproject vector to {projected_vector_path}')
    where_filter = f'{vector_field.upper()} IN (' + ', '.join([str(x) for x in field_value_list]) + ')'
    LOGGER.debug(f'{projected_vector_path} {where_filter}')
    geoprocessing.reproject_vector(
        vector_path, raster_projection_wkt, projected_vector_path,
        where_filter=where_filter, driver_name='GPKG')

    projected_vector_info = geoprocessing.get_vector_info(
        projected_vector_path)

    target_bb = geoprocessing.merge_bounding_box_list(
        [raster_info['bounding_box'], projected_vector_info['bounding_box']],
        'intersection')

    geoprocessing.warp_raster(
        raster_path, raster_info['pixel_size'], target_raster_path,
        'near', target_bb=target_bb,
        vector_mask_options={'mask_vector_path': projected_vector_path},
        working_dir=os.getcwd())
    LOGGER.info(f'all done, raster at {target_raster_path}')
    shutil.rmtree(temp_dir)


def main():
    parser = argparse.ArgumentParser(description='Global SWY')
    parser.add_argument(
        'scenario_config_path',
        help='Pattern to .INI file(s) that describes scenario(s) to run.')
    args = parser.parse_args()

    scenario_config_path_list = list(glob.glob(args.scenario_config_path))
    task_graph = taskgraph.TaskGraph(
        WORKSPACE_DIR,
        min(len(scenario_config_path_list), multiprocessing.cpu_count()), 5.0)
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
        if 'watershed_subset' in scenario_config:
            watershed_subset_payload = eval(
                scenario_config['watershed_subset'])
            if len(watershed_subset_payload) == 3:
                (watershed_basename, watershed_field,
                 watershed_field_id_list) = watershed_subset_payload
            clipped_dem = os.path.join(local_workspace, os.path.basename(
                scenario_config['dem_raster_path']))
            vector_path = os.path.join(
                scenario_config['watersheds_vector_path'],
                f'{watershed_basename}.shp')

            clip_raster_by_vector(
                scenario_config['dem_raster_path'],
                vector_path, watershed_field, watershed_field_id_list,
                clipped_dem)
            return
        # TODO: extract an AOI
        # TODO: clip DEM to AOI


        local_data_path_map = {
            'workspace_dir': scenario_config['workspace_dir'],
            'results_suffix': scenario_config['results_suffix'],
            'threshold_flow_accumulation': float(scenario_config['threshold_flow_accumulation']),
            'et0_dir': scenario_config['et0_dir'],
            'precip_dir': scenario_config['precip_dir'],
            'dem_raster_path': scenario_config['dem_raster_path'],
            'lulc_raster_path': scenario_config['lulc_raster_path'],
            'soil_group_path': scenario_config['soil_hydrologic_group_raster_path'],
            #'aoi_path': scenario_config['aoi_vector'],
            'biophysical_table_path': scenario_config['biophysical_table_path'],
            'rain_events_table_path': scenario_config['rain_events_table_path'],
            'monthly_alpha': False,
            'alpha_m': 1/12,
            'beta_i': scenario_config['beta_i'],
            'gamma': scenario_config['gamma'],
        }

        LOGGER.debug(local_data_path_map)


if __name__ == '__main__':
    main()
