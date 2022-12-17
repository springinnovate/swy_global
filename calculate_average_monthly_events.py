"""Calculate average monthly events over a given time period."""
import argparse
import os

import ee
import geemap
import geopandas
import requests

ERA5_RESOLUTION_M = 27830
ERA5_TOTAL_PRECIP_BAND_NAME = 'total_precipitation'

month_list = [
    '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
days_in_month_list = [
    '31', '28', '31', '30', '31', '30', '31', '31', '30', '31', '30', '31']


def main():
    parser = argparse.ArgumentParser(
        description='Monthly rain events by watershed in a yearly range.')
    parser.add_argument(
        'path_to_watersheds', help='Path to vector/shapefile of watersheds')
    parser.add_argument('start_year', type=int, help='start year YYYY')
    parser.add_argument('end_year', type=int, help='end year YYYY')
    parser.add_argument(
        '--authenticate', action='store_true',
        help='Pass this flag if you need to reauthenticate with GEE')
    parser.add_argument(
        '--rain_event_threshold', default=0.1, type=float,
        help='amount of rain (mm) in a day to count as a rain event')
    args = parser.parse_args()

    if args.authenticate:
        ee.Authenticate()
        return
    ee.Initialize()

    # convert to GEE polygon
    gp_poly = geopandas.read_file(args.path_to_watersheds).to_crs('EPSG:4326')
    local_shapefile_path = '_local_ok_to_delete.shp'
    gp_poly.to_file(local_shapefile_path)
    gp_poly = None
    ee_poly = geemap.shp_to_ee(local_shapefile_path)
    poly_mask = ee.Image.constant(1).clip(ee_poly).mask()

    base_era5_daily_collection = ee.ImageCollection("ECMWF/ERA5/DAILY")
    for month_val, day_in_month in zip(month_list, days_in_month_list):
        monthly_rain_event_image = ee.Image.constant(0).mask(poly_mask)
        for year in range(args.start_year, args.end_year+1):
            start_date = f'{year}-{month_val}-01'
            end_date = f'{year}-{month_val}-{day_in_month}'
            print(start_date, end_date)
            era5_month_collection = base_era5_daily_collection.filterDate(
                start_date, end_date)
            era5_daily_precip = era5_month_collection.select(
                ERA5_TOTAL_PRECIP_BAND_NAME).toBands().multiply(1000)  # convert to mm
            era5_daily_precip = era5_daily_precip.where(
                era5_daily_precip.lt(args.rain_event_threshold), 0).where(
                era5_daily_precip.gte(args.rain_event_threshold), 1)

            era5_precip_event_sum = era5_daily_precip.reduce('sum').clip(
                ee_poly).mask(poly_mask)
            monthly_rain_event_image = monthly_rain_event_image.add(
                era5_precip_event_sum)

        monthly_rain_event_image = monthly_rain_event_image.divide(
         args.end_year+1-args.start_year)
        url = monthly_rain_event_image.getDownloadUrl({
            'region': ee_poly.geometry().bounds(),
            'scale': ERA5_RESOLUTION_M,
            'format': 'GEO_TIFF'
        })
        response = requests.get(url)
        vector_basename = os.path.basename(os.path.splitext(args.path_to_watersheds)[0])
        precip_path = f"{vector_basename}_avg_precip_events_{args.start_year}_{args.end_year}_{month_val}_{args.rain_event_threshold}.tif"
        print(f'calculate total precip event {precip_path}')
        with open(precip_path, 'wb') as fd:
            fd.write(response.content)


if __name__ == '__main__':
    main()
