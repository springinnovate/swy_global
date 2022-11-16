"""Run SWY globally."""
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

    for _, _, hab_path in eval(scenario_config[HABITAT_MAP_KEY]).values():
        if not os.path.exists(hab_path):
            raise ValueError(
                f'expected a habitat raster at "{hab_path}" but one not found')

    return scenario_config, scenario_id


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
        LOGGER.debug(scenario_config)

    for scenario_config, scenario_id in config_scenario_list:
        habitat_map = eval(scenario_config[HABITAT_MAP_KEY])
        local_data_path_map = {
            'wwiii': scenario_config['WWIII_PATH'],
            'slr': scenario_config['SLR_PATH'],
            'geomorphology': scenario_config['GEOMORPHOLOGY_PATH'],
            'dem': scenario_config['DEM_PATH'],
            'lulc': scenario_config['LULC_PATH'],
            'global_polygon': scenario_config['GLOBAL_POLYGON_PATH'],
            'buffer_vector': scenario_config['BUFFER_VECTOR_PATH'],
            'shore_grid': scenario_config['SHORE_GRID_PATH'],
        }
        LOGGER.debug(habitat_map)
        LOGGER.debug(local_data_path_map)


if __name__ == '__main__':
    main()
