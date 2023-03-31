Seasonal Water Yield Computational Pipeline
===========================================

This project executes a variant of the InVEST Seasonal Water Yield model that allows for continuous biophysical data layers rather than the reliance of a landcover-to-lookup table based map. It is also a data pipeline that breaks large landscape analyses into smaller landscape components that can be processed in parallel. In short, use this project if you:

1) want to run the InVEST Seasonal Water Yield model on very large datasets (50k x 50k raster sizes or more), and/or,
2) have direct Earth Observation based data that you want to use instead of a landcover map (ex: a C factor raster built from an ML model of NDVI).

Configure the model run by defining necessary variables in an .ini file, examples located in ``pipeline_config_files``. For example, the following configuration file defines a run of SWY  on a single watershed where all curve number and Kc rasters are provided directly rather than being provided through a landcover map to lookup table. Any data not listed are defaulted in ``swy_global.ini`` but can be overridden.

    [wwf_ph_baseline_chirps_single_no_biophysical_table]
    PRECIP_DIR = D:\local_global_swy_data\precip_2010-2018_average
    ET0_DIR = D:\local_global_swy_data\Global-ET0_v3_monthly_tifs
    SOIL_HYDROLOGIC_GROUP_RASTER_PATH = D:\local_global_swy_data\HYSOGs250m_md5_517bfa.tif
    SOIL_HYDROLOGIC_MAP = {
      1: 'A',
      2: 'B',
      3: 'C',
      4: 'D',
      11: 'A',
      12: 'B',
      13: 'C',
      14: 'D'}

    USER_DEFINED_RAIN_EVENTS_PATH = ./ph_box_CHIRPS_avg_precip_events_2000_2010_*_1.0.tif
    MONTHLY_ALPHA = 0.018
    THRESHOLD_FLOW_ACCUMULATION = 1000
    WATERSHED_SUBSET = ('as_bas_15s_beta', 'BASIN_ID', 390065)

    ROOT_DEPTH_PATH = direct_data/root_depth_wwf_ph_baseline_chirps_single.tif
    CN_A_PATH = direct_data/CN_A_wwf_ph_baseline_chirps_single.tif
    CN_B_PATH = direct_data/CN_B_wwf_ph_baseline_chirps_single.tif
    CN_C_PATH = direct_data/CN_C_wwf_ph_baseline_chirps_single.tif
    CN_D_PATH = direct_data/CN_D_wwf_ph_baseline_chirps_single.tif
    KC_1_PATH = direct_data/kc_1_wwf_ph_baseline_chirps_single.tif
    KC_2_PATH = direct_data/kc_2_wwf_ph_baseline_chirps_single.tif
    KC_3_PATH = direct_data/kc_3_wwf_ph_baseline_chirps_single.tif
    KC_4_PATH = direct_data/kc_4_wwf_ph_baseline_chirps_single.tif
    KC_5_PATH = direct_data/kc_5_wwf_ph_baseline_chirps_single.tif
    KC_6_PATH = direct_data/kc_6_wwf_ph_baseline_chirps_single.tif
    KC_7_PATH = direct_data/kc_7_wwf_ph_baseline_chirps_single.tif
    KC_8_PATH = direct_data/kc_8_wwf_ph_baseline_chirps_single.tif
    KC_9_PATH = direct_data/kc_9_wwf_ph_baseline_chirps_single.tif
    KC_10_PATH = direct_data/kc_10_wwf_ph_baseline_chirps_single.tif
    KC_11_PATH = direct_data/kc_11_wwf_ph_baseline_chirps_single.tif
    KC_12_PATH = direct_data/kc_12_wwf_ph_baseline_chirps_single.tif

To run the model execute the main script and pass any number of ``.ini`` files at the command line, for example: ``python run_swy_global.py wwf_ph_baseline_chirps_single_no_biophysical_table.ini`` During execution log data are streamed to stdout.

Final results will be located in a directory called ``workspace_{basename of ini file}`` so the example above will be located at ``workspace_wwf_ph_baseline_chirps_single_no_biophysical_table``.

The InVEST user's guide for this model is comparable, except for the option to provide custom continuous raster data in leiu of a landcover/lookup table combination <https://storage.googleapis.com/releases.naturalcapitalproject.org/invest-userguide/latest/en/seasonal_water_yield.html>.
