import os
import ee
import pickle
import zipfile
import requests
import itertools
import numpy as np
from glob import glob
from osgeo import gdal
import geopandas as gpd
from datetime import datetime

from Codes.utils.system_ops import makedirs, copy_file
from Codes.utils.raster_ops import read_raster_arr_object, write_array_to_raster, mosaic_rasters, \
    clip_resample_reproject_raster, mask_raster_by_extent, sum_rasters, mean_rasters, filter_raster_on_threshold

# ee.Authenticate()

# # if ee.Authenticate() doesn't work or gives 'gcloud' error_mae go to the
# 'https://developers.google.com/earth-engine/guides/python_install' link and follow instructions for
# local/remote machine.
# In local machine, install gcloud (if not already installed) and follow the steps from the link to authenticate.
# For remote machine, https://www.youtube.com/watch?v=k-8qFh8EfFA this link was helpful for installing gcloud.
# Couldn't figure out how to authenticate in server pc, will try with new workstation
# If authenticated, no need to run the authentication process again. Just start from ee.initialize()

no_data_value = -9999
model_res = 0.02000000000000000736  # in deg, ~2.22 km
WestUS_raster = '../../Data_main/Compiled_data/reference_rasters/Western_US_refraster_2km.tif'


def download_from_url(download_dir, url_list):
    """
    Download Data from url.

    :param download_dir: File path to save downloaded data.
    :param url_list: A list of url/urls to download.

    :return:
    """
    makedirs([download_dir])
    for url in url_list:
        fname = url.rsplit('/', 1)[1]
        print(f'Downloading {fname}......')
        r = requests.get(url, allow_redirects=True)
        download_fname = os.path.join(download_dir, fname)
        open(download_fname, 'wb').write(r.content)
        print('Download complete')


def extract_data(zip_dir, out_dir, search_by='*.zip', rename_file=True):
    """
    Extract zipped file.

    :param zip_dir: File path of zipped file.
    :param out_dir: File path where data will be extracted.
    :param search_by: Keyword for searching files, default is '*.zip'.
    :param rename_file: True if file rename is required while extracting.

    :return: List of zipped files (can be used for deleting all the files). This function will unzip these files.
    """
    print('Extracting zip files.....')

    makedirs([out_dir])
    all_zipped_files = glob(os.path.join(zip_dir, search_by))

    for zip_file in all_zipped_files:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            if rename_file:
                zip_key = zip_file[zip_file.rfind(os.sep) + 1:zip_file.rfind('.')]
                zip_info = zip_ref.infolist()[0]
                zip_info.filename = zip_key + '.tif'
                zip_ref.extract(zip_info, path=out_dir)
            else:
                zip_ref.extractall(path=out_dir)
    return all_zipped_files


def get_gee_dict(data_name):
    ee.Initialize()
    gee_data_dict = {
        'SMAP_SM': 'NASA_USDA/HSL/SMAP10KM_soil_moisture',
        'LANDSAT_NDWI': 'LANDSAT/LC08/C01/T1_8DAY_NDWI',  # check for cloudcover
        'LANDSAT_NDVI': 'LANDSAT/LC08/C01/T1_8DAY_NDVI',  # check for cloudcover
        'GPM_PRECIP': 'NASA/GPM_L3/IMERG_MONTHLY_V06',
        'GRIDMET_PRECIP': 'IDAHO_EPSCOR/GRIDMET',
        'PRISM_PRECIP': 'OREGONSTATE/PRISM/AN81m',
        'VIIRS_NDVI': 'NOAA/VIIRS/001/VNP13A1',  # cloudcover mask added later
        'VIIRS_EVI': 'NOAA/VIIRS/001/VNP13A1',  # cloudcover mask added later
        'VIIRS_EVI2': 'NOAA/VIIRS/001/VNP13A1',  # cloudcover mask added later
        'MODIS_Day_LST': 'MODIS/006/MOD11A2',  # check for cloudcover
        'MODIS_Terra_NDVI': 'MODIS/006/MOD13Q1',  # cloudcover mask added later
        'MODIS_Terra_EVI': 'MODIS/006/MOD13Q1',  # cloudcover mask added later
        'MODIS_NDWI': 'MODIS/006/MOD09A1',  # cloudcover mask added later
        'MODIS_LAI': 'MODIS/006/MCD15A3H',
        'MODIS_ET': 'MODIS/006/MOD16A2',  # unit in kg/m2
        'TERRACLIMATE_ET': 'IDAHO_EPSCOR/TERRACLIMATE',
        'FAO_ACTUAL_ET': 'FAO/WAPOR/2/L1_AETI_D',
        'FAO_EVAPORATION': 'FAO/WAPOR/2/L1_E_D',
        'OpenET_ensemble': 'OpenET/ENSEMBLE/CONUS/GRIDMET/MONTHLY/v2_0',
        'FLDAS_BF_GW_RO': 'NASA/FLDAS/NOAH01/C/GL/M/V001',
        'FLDAS_SM': 'NASA/FLDAS/NOAH01/C/GL/M/V001',
        'USDA_CDL': 'USDA/NASS/CDL',  # Recommending not to use GEE. Download data from direct USDA link
        'GPW_Pop': 'CIESIN/GPWv411/GPW_UNWPP-Adjusted_Population_Density'
    }
    gee_band_dict = {
        'SMAP_SM': 'ssm',
        'LANDSAT_NDWI': 'NDWI',
        'LANDSAT_NDVI': 'NDVI',
        'GPM_PRECIP': 'precipitation',
        'GRIDMET_PRECIP': 'pr',
        'PRISM_PRECIP': 'ppt',
        'VIIRS_NDVI': 'NDVI',
        'VIIRS_EVI': 'EVI',
        'VIIRS_EVI2': 'EVI2',
        'MODIS_Day_LST': 'LST_Day_1km',
        'MODIS_Terra_NDVI': 'NDVI',
        'MODIS_Terra_EVI': 'EVI',
        'MODIS_NDWI': ['sur_refl_b02', 'sur_refl_b06'],  # bands for NIR and SWIR, respectively.
        'MODIS_LAI': 'Lai',
        'MODIS_ET': 'ET',
        'TERRACLIMATE_ET': 'aet',  # unit in mm, monthly total
        'FAO_ACTUAL_ET': 'L1_AETI_D',
        'FAO_EVAPORATION': 'L1_E_D',
        'OpenET_ensemble': 'et_ensemble_mad',  # unit in mm, monthly total
        'FLDAS_BF_GW_RO': 'Qsb_tavg',
        'FLDAS_SM': 'SoilMoi00_10cm_tavg',
        'USDA_CDL': 'cropland',
        'GPW_Pop': 'unwpp-adjusted_population_density'
    }
    gee_scale_dict = {
        'SMAP_SM': 1,
        'LANDSAT_NDWI': 1,
        'LANDSAT_NDVI': 1,
        'GPM_PRECIP': 1,
        'GRIDMET_PRECIP': 1,
        'PRISM_PRECIP': 1,
        'VIIRS_NDVI': 0.0001,
        'VIIRS_EVI': 0.0001,
        'VIIRS_EVI2': 0.0001,
        'MODIS_Day_LST': 0.02,
        'MODIS_Terra_NDVI': 0.0001,
        'MODIS_Terra_EVI': 0.0001,
        'MODIS_NDWI': 0.0001,
        'MODIS_LAI': 0.1,
        'MODIS_ET': 0.1,
        'TERRACLIMATE_ET': 0.1,
        'OpenET_ensemble': 1,
        'FAO_ACTUAL_ET': 0.1,
        'FAO_EVAPORATION': 0.1,
        'FLDAS_BF_GW_RO': 1,
        'FLDAS_SM': 1,
        'USDA_CDL': 1,
        'GPW_Pop': 1
    }

    aggregation_dict = {
        'SMAP_SM': ee.Reducer.sum(),
        'LANDSAT_NDWI': ee.Reducer.mean(),
        'LANDSAT_NDVI': ee.Reducer.mean(),
        'GPM_PRECIP': ee.Reducer.sum(),
        'GRIDMET_PRECIP': ee.Reducer.sum(),
        'PRISM_PRECIP': ee.Reducer.sum(),
        'VIIRS_NDVI': ee.Reducer.mean(),
        'VIIRS_EVI': ee.Reducer.mean(),
        'VIIRS_EVI2': ee.Reducer.mean(),
        'MODIS_Day_LST': ee.Reducer.mean(),
        'MODIS_Terra_NDVI': ee.Reducer.mean(),
        'MODIS_Terra_EVI': ee.Reducer.mean(),
        'MODIS_NDWI': ee.Reducer.mean(),
        'MODIS_LAI': ee.Reducer.mean(),
        'MODIS_ET': ee.Reducer.sum(),
        'TERRACLIMATE_ET': ee.Reducer.median(),
        # set for downloading monthly data, change if want to download yearly data
        'OpenET_ensemble': ee.Reducer.median(),
        # I am taking the median of 30m pixels to assign the value of 2km pixel. Change for yearly data download if needed.
        'FAO_ACTUAL_ET': ee.Reducer.sum(),
        'FAO_EVAPORATION': ee.Reducer.sum(),
        'FLDAS_BF_GW_RO': ee.Reducer.sum(),
        'FLDAS_SM': ee.Reducer.sum(),
        'USDA_CDL': ee.Reducer.first(),
        'GPW_Pop': ee.Reducer.first()
    }

    month_start_date_dict = {
        'SMAP_SM': datetime(2015, 4, 1),
        'LANDSAT_NDWI': datetime(2013, 4, 1),
        'LANDSAT_NDVI': datetime(2013, 4, 1),
        'GPM_PRECIP': datetime(2006, 6, 1),
        'GRIDMET_PRECIP': datetime(1979, 1, 1),
        'PRISM_PRECIP': datetime(1895, 1, 1),
        'VIIRS_NDVI': datetime(2012, 1, 1),
        'VIIRS_EVI': datetime(2012, 1, 1),
        'VIIRS_EVI2': datetime(2012, 1, 1),
        'MODIS_Day_LST': datetime(2000, 2, 1),
        'MODIS_Terra_NDVI': datetime(2000, 2, 1),
        'MODIS_Terra_EVI': datetime(2000, 2, 1),
        'MODIS_NDWI': datetime(2000, 2, 1),
        'MODIS_LAI': datetime(2002, 7, 1),
        'MODIS_ET': datetime(2001, 1, 1),
        'TERRACLIMATE_ET': datetime(1958, 1, 1),
        'OpenET_ensemble': datetime(2016, 1, 1),
        'FAO_ACTUAL_ET': datetime(2009, 1, 1),
        'FAO_EVAPORATION': datetime(2009, 1, 1),
        'FLDAS_BF_GW_RO': datetime(1982, 1, 1),
        'FLDAS_SM': datetime(1982, 1, 1),
        'USDA_CDL': datetime(1997, 1, 1),
        'GPW_Pop': datetime(2000, 1, 1)
    }

    month_end_date_dict = {
        'SMAP_SM': datetime(2022, 8, 2),
        'LANDSAT_NDWI': datetime(2022, 1, 1),
        'LANDSAT_NDVI': datetime(2022, 1, 1),
        'GPM_PRECIP': datetime(2021, 9, 1),
        'GRIDMET_PRECIP': datetime(2023, 9, 15),
        'PRISM_PRECIP': datetime(2023, 8, 1),
        'VIIRS_NDVI': datetime(2023, 8, 29),
        'VIIRS_EVI': datetime(2023, 8, 29),
        'VIIRS_EVI2': datetime(2023, 8, 29),
        'MODIS_Day_LST': datetime(2023, 8, 29),
        'MODIS_Terra_NDVI': datetime(2023, 8, 13),
        'MODIS_Terra_EVI': datetime(2023, 8, 13),
        'MODIS_NDWI': datetime(2023, 8, 29),
        'MODIS_LAI': datetime(2023, 9, 6),
        'MODIS_ET': datetime(2023, 8, 29),
        'TERRACLIMATE_ET': datetime(2022, 12, 1),
        'OpenET_ensemble': datetime(2022, 12, 1),
        'FAO_ACTUAL_ET': datetime(2023, 3, 1),
        'FAO_EVAPORATION': datetime(2023, 3, 1),
        'FLDAS_BF_GW_RO': datetime(2023, 7, 1),
        'FLDAS_SM': datetime(2023, 7, 1),
        'USDA_CDL': datetime(2022, 1, 1),
        'GPW_Pop': datetime(2020, 1, 1)
    }

    year_start_date_dict = {
        'SMAP_SM': datetime(2015, 1, 1),
        'LANDSAT_NDWI': datetime(2013, 1, 1),
        'LANDSAT_NDVI': datetime(2013, 1, 1),
        'GPM_PRECIP': datetime(2006, 1, 1),
        'GRIDMET_PRECIP': datetime(1979, 1, 1),
        'PRISM_PRECIP': datetime(1895, 1, 1),
        'VIIRS_NDVI': datetime(2012, 1, 1),
        'VIIRS_EVI': datetime(2012, 1, 1),
        'VIIRS_EVI2': datetime(2012, 1, 1),
        'MODIS_Day_LST': datetime(2000, 1, 1),
        'MODIS_Terra_NDVI': datetime(2000, 1, 1),
        'MODIS_Terra_EVI': datetime(2000, 1, 1),
        'MODIS_NDWI': datetime(2000, 1, 1),
        'MODIS_LAI': datetime(2002, 1, 1),
        'MODIS_ET': datetime(2001, 1, 1),
        'TERRACLIMATE_ET': datetime(1958, 1, 1),
        'OpenET_ensemble': datetime(2016, 1, 1),
        'FAO_ACTUAL_ET': datetime(2009, 1, 1),
        'FAO_EVAPORATION': datetime(2009, 1, 1),
        'FLDAS_BF_GW_RO': datetime(1982, 1, 1),
        'FLDAS_SM': datetime(1982, 1, 1),
        'USDA_CDL': datetime(1997, 1, 1),
        'GPW_Pop': datetime(2000, 1, 1)
    }

    year_end_date_dict = {
        'SMAP_SM': datetime(2023, 1, 1),
        'LANDSAT_NDWI': datetime(2022, 1, 1),
        'LANDSAT_NDVI': datetime(2022, 1, 1),
        'GPM_PRECIP': datetime(2022, 1, 1),
        'GRIDMET_PRECIP': datetime(2024, 1, 1),
        'PRISM_PRECIP': datetime(2024, 1, 1),
        'VIIRS_NDVI': datetime(2024, 1, 1),
        'VIIRS_EVI': datetime(2024, 1, 1),
        'VIIRS_EVI2': datetime(2024, 1, 1),
        'MODIS_Day_LST': datetime(2024, 1, 1),
        'MODIS_Terra_NDVI': datetime(2024, 1, 1),
        'MODIS_Terra_EVI': datetime(2024, 1, 1),
        'MODIS_NDWI': datetime(2024, 1, 1),
        'MODIS_LAI': datetime(2024, 1, 1),
        'MODIS_ET': datetime(2024, 1, 1),
        'TERRACLIMATE_ET': datetime(2023, 1, 1),
        'OpenET_ensemble': datetime(2023, 1, 1),
        'FAO_ACTUAL_ET': datetime(2024, 1, 1),
        'FAO_EVAPORATION': datetime(2024, 1, 1),
        'FLDAS_BF_GW_RO': datetime(2024, 1, 1),
        'FLDAS_SM': datetime(2024, 1, 1),
        'USDA_CDL': datetime(2022, 1, 1),
        'GPW_Pop': datetime(2020, 1, 1)
    }

    return gee_data_dict[data_name], gee_band_dict[data_name], gee_scale_dict[data_name], aggregation_dict[data_name], \
           month_start_date_dict[data_name], month_end_date_dict[data_name], year_start_date_dict[data_name], \
           year_end_date_dict[data_name],


def cloud_cover_filter(data_name, start_date, end_date, from_bit, to_bit, geometry_bounds):
    """
    Applies cloud cover mask on GEE data.

    :param data_name: Data Name.
           Valid dataset include- ['MODIS_Terra_NDVI', 'MODIS_Terra_EVI', 'MODIS_NDWI', 'VIIRS_NDVI',
           'VIIRS_EVI', 'VIIRS_EVI2'].
    :param start_date: Start date of data to download. Generated from download_gee_data() func.
    :param end_date: End date of data to download. Generated from download_gee_data() func.
    :param from_bit: Start bit to consider for masking.
    :param to_bit: End bit to consider for masking.
    :param geometry_bounds: GEE geometry object.

    :return: Cloud filtered imagecollection.
    """

    def bitwise_extract(img):
        """
        Applies cloudmask on image.
        :param Image.
        :return Cloud-masked image.
        """
        ee.Initialize()
        global qc_img
        if data_name in ('MODIS_Terra_NDVI', 'MODIS_Terra_EVI'):
            qc_img = img.select('DetailedQA')
        elif data_name in ('VIIRS_NDVI', 'VIIRS_EVI', 'VIIRS_EVI2'):
            qc_img = img.select('VI_Quality')
        elif data_name == 'MODIS_NDWI':
            qc_img = img.select('StateQA')

        masksize = ee.Number(1).add(to_bit).subtract(from_bit)
        mask = ee.Number(1).leftShift(masksize).subtract(1)
        apply_mask = qc_img.rightShift(from_bit).bitwiseAnd(mask).lte(1)
        return img.updateMask(apply_mask)

    if data_name in ('MODIS_Terra_NDVI', 'MODIS_Terra_EVI'):
        # filterBounds are not necessary, added it to reduce processing extent
        image = ee.ImageCollection('MODIS/061/MOD13Q1').filterDate(start_date, end_date).filterBounds(geometry_bounds)
        cloud_masked = image.map(bitwise_extract)
        return cloud_masked

    elif data_name in ('VIIRS_NDVI', 'VIIRS_EVI', 'VIIRS_EVI2'):
        image = ee.ImageCollection('NOAA/VIIRS/001/VNP13A1').filterDate(start_date, end_date). \
            filterBounds(geometry_bounds)
        cloud_masked = image.map(bitwise_extract)
        return cloud_masked

    elif data_name == 'MODIS_NDWI':
        image = ee.ImageCollection('MODIS/061/MOD09A1').filterDate(start_date, end_date).filterBounds(geometry_bounds)
        cloud_masked = image.map(bitwise_extract)
        return cloud_masked


# # The download_gee_data_for_grow_season() function isn't fully optimized. Might need to change things at the
# #  get_gee_dict() function, Not using this function for the current project
# def download_gee_data_for_grow_season \
#                 (data_name, download_dir, year_list, merge_keyword, grid_shape, refraster=WestUS_raster):
#     """
#     process and download data from GEE based on varying growing season. This code will mosaic data for different growing
#     season together. So there will be visible difference in data. Consider smoothening the grids used for downloading.
#
#     **Saving as a reference code. Will not use in this project. Rather data will be downloaded for each month
#     (for selected years) and then merged together based on growing season to make the data smoother.
#
#     :param data_name: Data name.
#     Current valid data names are -
#         ['SMAP_SM', 'LANDSAT_NDWI', 'LANDSAT_NDVI', 'GPM_PRECIP', 'GRIDMET_PRECIP', 'PRISM_PRECIP', 'VIIRS_NDVI',
#         'VIIRS_EVI', 'VIIRS_EVI2', 'MODIS_Day_LST', 'MODIS_Terra_NDVI', 'MODIS_Terra_EVI', 'MODIS_NDWI', 'MODIS_LAI',
#         'MODIS_ET', 'FAO_ACTUAL_ET', 'FAO_EVAPORATION', 'FLDAS_BF_GW_RO', 'FLDAS_SM',]
#     :param download_dir: File path of download directory.
#     :param year_list: List of years to download data for, e.g. [2000, 2005, 2010, 2015].
#     :param merge_keyword: Keyword to use for merging downloaded data. Suggested 'WestUS'/'Conus'.
#     :param grid_shape: File path of grid shape for which data will be downloaded and mosaiced.
#     :param refraster: Reference raster to use for merging downloaded datasets.
#
#     :return: File path of downloaded raster data.
#     """
#     ee.Initialize()
#     makedirs([download_dir])
#
#     downloaded_raster_dict = {}
#
#     grow_season_dict = {'Apr01-Oct31': [4, 10], 'Jan01-Dec31': [1, 12], 'Jul01-Aug31': [7, 8], 'Mar01-Nov30': [3, 11],
#                         'May01-Sep30': [5, 9]}  # A dictionary for assigning growing season months
#
#     for year in year_list:
#         grids = gpd.read_file(grid_shape)
#         grids = grids.sort_values(by='grid_no', ascending=True)
#         growing_season = grids['GrowSeason']
#         grid_geometry = grids['geometry']
#         grid_no = grids['grid_no']
#
#         start_year, end_year, start_month, end_month = None, None, None, None
#
#         for grid, grow_period, serial in zip(grid_geometry, growing_season, grid_no):
#             roi = grid.bounds
#             gee_extent = ee.Geometry.Rectangle(roi)
#
#             # Selecting month ranges based on grid files' growing season periods
#             if grow_period == 'Apr01-Oct31':
#                 start_year = year
#                 end_year = year
#                 start_month, end_month = grow_season_dict['Apr01-Oct31'][0], grow_season_dict['Apr01-Oct31'][1]
#
#             elif grow_period == 'Jan01-Dec31':
#                 start_year = year
#                 end_year = year + 1
#                 start_month, end_month = grow_season_dict['Jan01-Dec31'][0], 1  # up to January 1st of next year
#
#             elif grow_period == 'Jul01-Aug31':
#                 start_year = year
#                 end_year = year
#                 start_month, end_month = grow_season_dict['Jul01-Aug31'][0], grow_season_dict['Jul01-Aug31'][1]
#
#             elif grow_period == 'Mar01-Nov30':
#                 start_year = year
#                 end_year = year
#                 start_month, end_month = grow_season_dict['Mar01-Nov30'][0], grow_season_dict['Mar01-Nov30'][1]
#
#             elif grow_period == 'May01-Sep30':
#                 start_year = year
#                 end_year = year
#                 start_month, end_month = grow_season_dict['May01-Sep30'][0], grow_season_dict['May01-Sep30'][1]
#
#             start_date = ee.Date.fromYMD(start_year, start_month, 1)
#             end_date = ee.Date.fromYMD(end_year, end_month + 1, 1)
#             data, band, multiply_scale, reducer, month_start_range, month_end_range, \
#             year_start_range, year_end_range = get_gee_dict(data_name)
#
#             if data_name in ('MODIS_Terra_NDVI', 'MODIS_Terra_EVI'):
#                 download_data = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent).select(band). \
#                     reduce(reducer).multiply(multiply_scale).toFloat()
#
#             elif data_name in ('VIIRS_NDVI', 'VIIRS_EVI', 'VIIRS_EVI2'):
#                 download_data = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent).select(band). \
#                     reduce(reducer).multiply(multiply_scale).toFloat()
#
#             elif data_name == 'MODIS_NDWI':
#                 nir = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent).select(band[0]) \
#                     .reduce(reducer).multiply(multiply_scale).toFloat()
#                 swir = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent).select(band[1]) \
#                     .reduce(reducer).multiply(multiply_scale).toFloat()
#                 download_data = nir.subtract(swir).divide(nir.add(swir))
#             elif data_name == 'GPW_Pop':
#                 start_date = ee.Date.fromYMD(year, 1, 1)  # GPW population dataset's data starts at
#                 end_date = ee.Date.fromYMD(year, 12, 31)
#                 download_data = ee.ImageCollection(data).select(band).filterDate(start_date, end_date). \
#                     filterBounds(gee_extent).reduce(reducer).toFloat()
#             else:
#                 download_data = ee.ImageCollection(data).select(band).filterDate(start_date, end_date). \
#                     filterBounds(gee_extent).reduce(reducer).multiply(multiply_scale).toFloat()
#
#             data_url = download_data.getDownloadURL({'name': data_name,
#                                                      'crs': 'EPSG:4269',  # NAD83
#                                                      'scale': 2000,  # in meter. equal to ~0.02 deg
#                                                      'region': gee_extent,
#                                                      'format': 'GEO_TIFF'})
#
#             key_word = data_name
#             local_file_name = os.path.join(download_dir, f'{key_word}_{str(year)}_{str(serial)}.tif')
#             print('Downloading', local_file_name, '.....')
#             r = requests.get(data_url, allow_redirects=True)
#             open(local_file_name, 'wb').write(r.content)
#
#         mosaic_name = f'{data_name}_{year}.tif'
#         mosaic_dir = os.path.join(download_dir, f'{merge_keyword}')
#         makedirs([mosaic_dir])
#         downloaded_arr, downloaded_raster = mosaic_rasters(download_dir, mosaic_dir, mosaic_name, ref_raster=refraster,
#                                                            search_by=f'*{year}*.tif', nodata=no_data_value)
#         print('Downloaded Data Merged')
#         downloaded_raster_dict[mosaic_name] = downloaded_raster
#
#     return downloaded_raster_dict


# # The download_gee_data_yearly() function isn't fully optimized. Might need to change things at the
# #  get_gee_dict() function, Not using this function for the current project
def download_gee_data_yearly(data_name, download_dir, year_list, month_range, merge_keyword, grid_shape,
                             refraster=WestUS_raster):
    """
    Download data (at yearly scale) from GEE.

    :param data_name: Data name.
    Current valid data names are -
        ['SMAP_SM', 'LANDSAT_NDWI', 'LANDSAT_NDVI', 'GPM_PRECIP', 'GRIDMET_PRECIP', 'PRISM_PRECIP', 'VIIRS_NDVI',
        'VIIRS_EVI', 'VIIRS_EVI2', 'MODIS_Day_LST', 'MODIS_Terra_NDVI', 'MODIS_Terra_EVI', 'MODIS_NDWI', 'MODIS_LAI',
        'MODIS_ET', 'FAO_ACTUAL_ET', 'FAO_EVAPORATION', 'FLDAS_BF_GW_RO', 'FLDAS_SM']
    :param download_dir: File path of download directory.
    :param year_list: List of years to download data for.
    :param month_range: Tuple of month ranges to download data for, e.g., for months 4-12 use (4, 12).
    :param merge_keyword: Keyword to use for merging downloaded data. Suggested 'WestUS'/'Conus'.
    :param grid_shape: File path of grid shape for which data will be downloaded and mosaiced.
    :param refraster: Reference raster to use for merging downloaded datasets.

    :return: File path of downloaded raster data.
    """
    ee.Initialize()
    download_dir = os.path.join(download_dir, data_name)
    makedirs([download_dir])

    # Extracting dataset information required for downloading from GEE
    data, band, multiply_scale, reducer, month_start_range, month_end_range, \
    year_start_range, year_end_range = get_gee_dict(data_name)

    # Loading grid files to be used for data download
    grids = gpd.read_file(grid_shape)
    grids = grids.sort_values(by='grid_no', ascending=True)
    grid_geometry = grids['geometry']
    grid_no = grids['grid_no']

    for year in year_list:  # first loop for years
        start_date = ee.Date.fromYMD(year, month_range[0], 1)
        start_date_dt = datetime(year, month_range[0], 1)
        if month_range[1] < 12:
            end_date = ee.Date.fromYMD(year, month_range[1] + 1, 1)
            end_date_dt = datetime(year, month_range[1] + 1, 1)
        else:
            end_date = ee.Date.fromYMD(year + 1, 1, 1)  # for month 12 moving end date to next year
            end_date_dt = datetime(year + 1, 1, 1)

        # a condition to check whether start and end date falls in the available data range in GEE
        # if not the block will not be executed
        if (start_date_dt >= year_start_range) & (end_date_dt <= year_end_range):

            for grid_sr, geometry in zip(grid_no, grid_geometry):  # second loop for grids
                roi = geometry.bounds
                gee_extent = ee.Geometry.Rectangle(roi)

                # Filtering/processing datasets with data ranges, cloudcover, geometry, band, reducer, scale
                if data_name in ('MODIS_Terra_NDVI', 'MODIS_Terra_EVI'):
                    download_data = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent).select(band). \
                        reduce(reducer). \
                        multiply(multiply_scale).toFloat()

                elif data_name in ('VIIRS_NDVI', 'VIIRS_EVI', 'VIIRS_EVI2'):
                    download_data = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent).select(band). \
                        reduce(reducer).multiply(multiply_scale).toFloat()

                elif data_name == 'MODIS_NDWI':
                    nir = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent).select(band[0]). \
                        reduce(reducer).multiply(multiply_scale).toFloat()
                    swir = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent).select(band[1]). \
                        reduce(reducer).multiply(multiply_scale).toFloat()
                    download_data = nir.subtract(swir).divide(nir.add(swir))
                elif data_name == 'GPW_Pop':
                    start_date = ee.Date.fromYMD(year, 1, 1)  # GPW population dataset's data starts at
                    end_date = ee.Date.fromYMD(year, 12, 31)
                    # filterBounds are not necessary, added it to reduce processing extent
                    download_data = ee.ImageCollection(data).select(band).filterDate(start_date, end_date). \
                        filterBounds(gee_extent).reduce(reducer).toFloat()
                else:
                    download_data = ee.ImageCollection(data).select(band).filterDate(start_date, end_date). \
                        filterBounds(gee_extent).reduce(reducer).multiply(multiply_scale).toFloat()

                data_url = download_data.getDownloadURL({'name': data_name,
                                                         'crs': 'EPSG:4269',  # NAD83
                                                         'scale': 2000,  # in meter. equal to ~0.02 deg
                                                         'region': gee_extent,
                                                         'format': 'GEO_TIFF'})
                key_word = data_name
                local_file_name = os.path.join(download_dir, f'{key_word}_{str(year)}_{str(grid_sr)}.tif')
                print('Downloading', local_file_name, '.....')
                r = requests.get(data_url, allow_redirects=True)
                open(local_file_name, 'wb').write(r.content)

            mosaic_name = f'{data_name}_{year}.tif'
            mosaic_dir = os.path.join(download_dir, f'{merge_keyword}')
            makedirs([mosaic_dir])
            mosaic_rasters(download_dir, mosaic_dir, mosaic_name, ref_raster=refraster, search_by=f'*{year}*.tif',
                           nodata=no_data_value)
            print(f'{data_name} yearly data downloaded and merged')

        else:
            pass


def download_gee_data_monthly(data_name, download_dir, year_list, month_range, merge_keyword, grid_shape,
                              refraster=WestUS_raster):
    """
    Download data (at monthly scale) from GEE.

    :param data_name: Data name.
    Current valid data names are -
        ['SMAP_SM', 'LANDSAT_NDWI', 'LANDSAT_NDVI', 'GPM_PRECIP', 'GRIDMET_PRECIP', 'PRISM_PRECIP', 'VIIRS_NDVI',
        'VIIRS_EVI', 'VIIRS_EVI2', 'MODIS_Day_LST', 'MODIS_Terra_NDVI', 'MODIS_Terra_EVI', 'MODIS_NDWI', 'MODIS_LAI',
        'MODIS_ET', 'FAO_ACTUAL_ET', 'FAO_EVAPORATION', 'FLDAS_BF_GW_RO', 'FLDAS_SM', 'TERRSCLIMATE_ET']
    :param download_dir: File path of download directory.
    :param year_list: List of years to download data for.
    :param month_range: Tuple of month ranges to download data for, e.g., for months 1-12 use (1, 12).
    :param merge_keyword: Keyword to use for merging downloaded data. Suggested 'WestUS'/'Conus'.
    :param grid_shape: File path of grid shape for which data will be downloaded and mosaiced.
    :param refraster: Reference raster to use for merging downloaded datasets.

    :return: File path of downloaded raster data.
    """
    ee.Initialize()
    download_dir = os.path.join(download_dir, data_name)
    makedirs([download_dir])

    # Extracting dataset information required for downloading from GEE
    data, band, multiply_scale, reducer, month_start_range, month_end_range, \
    year_start_range, year_end_range = get_gee_dict(data_name)

    # Loading grid files to be used for data download
    grids = gpd.read_file(grid_shape)
    grids = grids.sort_values(by='grid_no', ascending=True)
    grid_geometry = grids['geometry']
    grid_no = grids['grid_no']

    month_list = [m for m in range(month_range[0], month_range[1] + 1)]  # creating list of months
    for year in year_list:  # first loop for years
        for month in month_list:  # second loop for months
            start_date = ee.Date.fromYMD(year, month, 1)
            start_date_dt = datetime(year, month, 1)

            if month < 12:
                end_date = ee.Date.fromYMD(year, month + 1, 1)
                end_date_dt = datetime(year, month + 1, 1)

            else:
                end_date = ee.Date.fromYMD(year + 1, 1, 1)  # for month 12 moving end date to next year
                end_date_dt = datetime(year + 1, 1, 1)

            # a condition to check whether start and end date falls in the available data range in GEE
            # if not the block will not be executed
            if (start_date_dt >= month_start_range) & (end_date_dt <= month_end_range):

                for grid_sr, geometry in zip(grid_no, grid_geometry):  # third loop for grids
                    roi = geometry.bounds
                    gee_extent = ee.Geometry.Rectangle(roi)

                    # Filtering/processing datasets with data ranges, cloudcover, geometry, band, reducer, scale
                    if data_name in ('MODIS_Terra_NDVI', 'MODIS_Terra_EVI'):
                        download_data = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent).select(
                            band). \
                            reduce(reducer). \
                            multiply(multiply_scale).toFloat()

                    elif data_name in ('VIIRS_NDVI', 'VIIRS_EVI', 'VIIRS_EVI2'):
                        download_data = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent).select(
                            band). \
                            reduce(reducer).multiply(multiply_scale).toFloat()

                    elif data_name == 'MODIS_NDWI':
                        nir = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent).select(band[0]). \
                            reduce(reducer).multiply(multiply_scale).toFloat()
                        swir = cloud_cover_filter(data_name, start_date, end_date, 0, 1, gee_extent).select(band[1]). \
                            reduce(reducer).multiply(multiply_scale).toFloat()
                        download_data = nir.subtract(swir).divide(nir.add(swir))

                    elif data_name in ('TERRACLIMATE_ET', 'OpenET_ensemble'):
                        download_data = ee.ImageCollection(data).select(band).filterDate(start_date, end_date). \
                            filterBounds(gee_extent).reduce(reducer).multiply(multiply_scale).toFloat()

                    elif data_name == 'GPW_Pop':
                        start_date = ee.Date.fromYMD(year, 1, 1)  # GPW population dataset's data starts at
                        end_date = ee.Date.fromYMD(year, 12, 31)
                        # filterBounds are not necessary, added it to reduce processing extent
                        download_data = ee.ImageCollection(data).select(band).filterDate(start_date, end_date). \
                            filterBounds(gee_extent).reduce(reducer).toFloat()

                    else:
                        download_data = ee.ImageCollection(data).select(band).filterDate(start_date, end_date). \
                            filterBounds(gee_extent).reduce(reducer).multiply(multiply_scale).toFloat()

                    data_url = download_data.getDownloadURL({'name': data_name,
                                                             'crs': 'EPSG:4269',  # NAD83
                                                             'scale': 2000,  # in meter. equal to ~0.02 deg
                                                             'region': gee_extent,
                                                             'format': 'GEO_TIFF'})
                    key_word = data_name
                    local_file_name = os.path.join(download_dir,
                                                   f'{key_word}_{str(year)}_{str(month)}_{str(grid_sr)}.tif')
                    print('Downloading', local_file_name, '.....')
                    r = requests.get(data_url, allow_redirects=True)
                    open(local_file_name, 'wb').write(r.content)

                    # # This is a check block to see if downloaded datasets are OK
                    # # sometimes a particular grid's data is corrupted but it's a random thing
                    # # not sure why it happens. Re-downloading the same data might not have that error
                    try:
                        arr = read_raster_arr_object(local_file_name, get_file=False)

                    except:
                        print('Downloaded data corrupted. Re-downloading.....')
                        print('Downloading', local_file_name, '.....')
                        r = requests.get(data_url, allow_redirects=True)
                        open(local_file_name, 'wb').write(r.content)

                mosaic_name = f'{data_name}_{year}_{month}.tif'
                mosaic_dir = os.path.join(download_dir, f'{merge_keyword}')
                makedirs([mosaic_dir])
                search_by = f'*{year}_{month}*.tif'
                mosaic_rasters(download_dir, mosaic_dir, mosaic_name, ref_raster=refraster, search_by=search_by,
                               nodata=no_data_value)
                print(f'{data_name} monthly data downloaded and merged')

            else:
                pass


def download_all_gee_data_monthly(data_list, download_dir, year_list, month_range, grid_shape,
                                  refraster=WestUS_raster, skip_download=False):
    """
    Used to download all gee data together.

    :param data_list: List of valid data names to download.
    Current valid data names are -
        ['SMAP_SM', 'LANDSAT_NDWI', 'LANDSAT_NDVI', 'GPM_PRECIP', 'GRIDMET_PRECIP', 'PRISM_PRECIP', 'VIIRS_NDVI',
        'VIIRS_EVI', 'VIIRS_EVI2', 'MODIS_Day_LST', 'MODIS_Terra_NDVI', 'MODIS_Terra_EVI', 'MODIS_NDWI', 'MODIS_LAI',
        'MODIS_ET', 'FAO_ACTUAL_ET', 'FAO_EVAPORATION', 'FLDAS_BF_GW_RO', 'FLDAS_SM', 'TERRACLIMATE_ET']
    :param download_dir: File path of main download directory. It will consist directory of individual dataset.
    :param year_list: List of years to download data for.
    :param month_range: Tuple of month ranges to download data for, e.g., for months 1-12 use (1, 12).
    :param grid_shape: File path of grid shape for which data will be downloaded and mosaiced.
    :param refraster: Reference raster to use for merging downloaded datasets.
    :param skip_download: Set to True to skip download.

    :return: None
    """
    if not skip_download:
        for data_name in data_list:
            data_download_dir = os.path.join(download_dir, data_name)
            makedirs([data_download_dir])

            # for datasets that needed to be downloaded on monthly scale
            download_gee_data_monthly(data_name=data_name, download_dir=download_dir, year_list=year_list,
                                      month_range=month_range, merge_keyword='WestUS_monthly',
                                      refraster=refraster, grid_shape=grid_shape)
    else:
        pass


def download_ssebop_et(years_list, month_range_list, download_dir='../../Data_main/Raster_data/Ssebop_ETa',
                       ssebop_link='https://edcintl.cr.usgs.gov/downloads/sciweb1/shared/uswem/web/conus/eta/modis_eta/monthly/downloads/',
                       skip_download=False):
    """
    Download ssebop actual ET data (unit in mm).

    :param years_list: List of years for which to download data.
    :param month_range_list: List of first and last month, i.e., to download data from April-September use [4, 9].
    :param download_dir: Directory path to download data.
    :param ssebop_link: USGS link with ssebop ET data.
    :param skip_download: Set to True to skip download.

    :return: None.
    """
    makedirs([download_dir])
    if not skip_download:
        months_list = [m for m in range(month_range_list[0], month_range_list[1] + 1)]

        year_month_combs = list(itertools.product(years_list, months_list))

        for each in year_month_combs:
            year, month = each
            print(f'Downloading SSEBOP ET for year {year} month {month}...')

            if len(str(month)) == 1:
                ssebop_et_link = f'{ssebop_link}m{str(year)}0{str(month)}.zip'
            else:
                ssebop_et_link = f'{ssebop_link}m{str(year)}{str(month)}.zip'
            r = requests.get(ssebop_et_link, allow_redirects=True)

            download_name = ssebop_et_link[ssebop_et_link.rfind('/') + 1:]
            download_to = os.path.join(download_dir, download_name)

            open(download_to, 'wb').write(r.content)

        zipped_files = extract_data(download_dir, download_dir, search_by='*.zip', rename_file=True)
        for z in zipped_files:
            os.remove(z)


def download_all_datasets(year_list, month_range, grid_shape, gee_data_list=None,
                          data_download_dir='../../Data_main/Raster_data',
                          download_gee_data_monthly_scale=True, skip_download_gee_data=True,
                          skip_download_ssebop_data=True):
    """
    Download all GEE datasets and ssebop data.

    :param year_list: List of years to download data for. We will use data for [2010, 2015] in the model.
    :param month_range: Tuple of month ranges to download data for, e.g., for months 1-12 use (1, 12).
    :param grid_shape: File path of grid shape for which data will be downloaded and mosaiced.
    :param gee_data_list: List of data to download from GEE. Default set to None, use if skip_download_gee_data=True.
                          Datasets currently used in the model:
                          ['MODIS_NDWI', 'GPM_PRECIP', 'MODIS_Day_LST', 'PRISM_PRECIP']
    :param data_download_dir: Directory path to download and save data.
    :param download_gee_data_monthly_scale: If GEE data needs to be downloaded at monthly scale.
                                            Default set to True for downloading monthly data.
                                            Set False to download data at yearly scale for a range of months.
    :param skip_download_gee_data: Set to False if want to download listed data. Default set to True.
    :param skip_download_ssebop_data: Set to False if want to download ssebop data. Default set to True.

    :return: None.
    """
    download_all_gee_data_monthly(gee_data_list, download_dir=data_download_dir,
                                  year_list=year_list, month_range=month_range, refraster=WestUS_raster,
                                  grid_shape=grid_shape,
                                  skip_download=skip_download_gee_data)

    ssebop_download_dir = os.path.join(data_download_dir, 'Ssebop_ETa')
    download_ssebop_et(year_list, month_range_list=[1, 12],
                       download_dir=ssebop_download_dir,
                       skip_download=skip_download_ssebop_data)


def process_cdl_data_to_cropland(cdl_dir='../../Data_main/Raster_data/USDA_CDL/CONUS_original_30m',
                                 processing_shp_dir='../../Data_main/Raster_data/USDA_CDL/processing_shapes',
                                 westus_30m_dir='../../Data_main/Raster_data/USDA_CDL/WestUS_30m',
                                 westus_processed_cropland_dir='../../Data_main/Raster_data/USDA_CDL/WestUS_Processed_cropland',
                                 ref_raster=WestUS_raster, resolution=model_res,
                                 already_masked=False, try_maximum_occurrence_approach=False,
                                 skip_processing=False):
    """
    Mask and process cdl datasets to only cropland. Processing includes selecting only cropland pixels and resampling
    to 2km resolution.

    CDL data download link : https://www.nass.usda.gov/Research_and_Science/Cropland/Release/

    :param cdl_dir: Directory path of original (CONUS scale) CDL datasets.
    :param processing_shp_dir: Directory path of shapefiles that is used to process cdl data. cdl datasets are 30m
                               resolution and memory-heavy. These shapefiles are used to mask the cdl datasets,
                               which is later processed individually.
    :param westus_30m_dir: Output directory path for saving Western US scale 30m cdl data.
                           Also, automatically saves intermediate datasets before final cropland products.
    :param westus_processed_cropland_dir: Output directory path for saving finalized Western US scale 2km cropland data.

    :param ref_raster: Filepath of Western US reference raster (set to default).
    :param resolution: Model resolution. Default set to 0.02000000000000000736.
    :param already_masked: Set to True to skip masking (masked cdl data has already been prepared).
    :param try_maximum_occurrence_approach: Defaults to False to set 2008 dataset value as 2000 & 2005 datasets.
                                            'try_maximum_occurrence_approach': (set to True to use)
                                            2000 and 2005 cdl datasets for whole Western US isn't available.
                                            This approach sets the maximum occurring cropland/non-cropland values from
                                            2008-2011 as 2000 & 2005 datasets.
    :param skip_processing: Set to True to skip cdl cropland dataset processing.

    :return: None.
    """
    if not skip_processing:
        print('processing cdl data to cropland...')
        cdl_US = glob(os.path.join(cdl_dir, '*.tif'))
        processing_bounds = glob(os.path.join(processing_shp_dir, '*.shp'))

        cdl_2008_arr, cdl_2009_arr, cdl_2010_arr, cdl_2011_arr, cdl_file, shape = None, None, None, None, None, None
        if not already_masked:
            # Loop for masking cdl raster to Western US extent.
            # Using 4 bounding boxes to mask and split the cdl for Western US into 4 regions for each year
            # to minimize memory load.
            for cdl_ras in cdl_US:
                for shape in processing_bounds:
                    raster_name = os.path.splitext(os.path.basename(cdl_ras))[0]
                    shape_name = os.path.splitext(os.path.basename(shape))[0]
                    name = raster_name + '_' + shape_name + '.tif'
                    mask_raster_by_extent(input_raster=cdl_ras, ref_file=shape, output_dir=westus_30m_dir,
                                          raster_name=name, invert=False, crop=True, nodata=0)

        # Creating new directory inside westus_30m_dir to save processed cropland data
        # Creating new directory to save processed cropland data
        cropland_30m_dir = os.path.join(westus_30m_dir, 'cropland')
        makedirs([cropland_30m_dir, westus_processed_cropland_dir])

        # converting to crop vs non-crop and developed vs non-developed data
        noncrop_classes = [0, 58, 59, 60, 61, 63, 64, 65, 81, 82, 83, 87, 88, 111, 121, 122, 123, 124, 131, 141, 142,
                           143, 152, 176, 190, 195]  # 0 is no data value

        # Listing all 30m cdl datasets for WestUS (each year consists of 4 cdl datasets, each for a particular region)
        cdl_westus = glob(os.path.join(westus_30m_dir, '*.tif'))

        # cropped each year's cdl data in 4 shapefile bounds for memory issue. They are specified as 'bound1', 'bound2',...
        # at the end of each raster's name. In this tep, we will select only the cropland pixels, resample them to 2km.
        for cdl_ras in cdl_westus:
            # crop vs non-crop raster  # if array value in noncrop list assigns 0, otherwise assigns 1
            # cdl data has no_data value set to 0
            cdl_arr, cdl_file = read_raster_arr_object(cdl_ras, change_dtype=False)  # datatype unsigned integer 8bit
            crop_arr = np.where(~np.isin(cdl_arr, noncrop_classes), 1, 0)

            crop_raster_name = os.path.splitext(os.path.basename(cdl_ras))[0] + '_crop.tif'
            crop_raster_path = os.path.join(cropland_30m_dir, crop_raster_name)

            # saving 30m cropland rasters
            # setting nodata to None, otherwise all 0 values will become nodata.
            # nodata pixels will be assigned to -9999 at mosaicing step
            crop_30m_raster = write_array_to_raster(raster_arr=crop_arr, raster_file=cdl_file,
                                                    transform=cdl_file.transform, output_path=crop_raster_path,
                                                    nodata=None, dtype=np.uint8)

            # Resampling 30m raster to 2km using 'average' resampling algorithm
            # the values represent percent (weighted average) cropland in the pixel
            year = crop_raster_name.split('_')[0]
            bound_no = crop_raster_name.split('_')[3]
            resampled_raster_name = f'{year}_2km_crop_percent_{bound_no}.tif'
            resampled_raster = clip_resample_reproject_raster(input_raster=crop_30m_raster, input_shape=None,
                                                              raster_name=resampled_raster_name,
                                                              output_raster_dir=cropland_30m_dir,
                                                              clip=False, resample=True, clip_and_resample=False,
                                                              targetaligned=True, resample_algorithm='average',
                                                              resolution=resolution, crs='EPSG:4269',
                                                              output_datatype=gdal.GDT_Float32)

            # filtering resampled rasters by a value threshold and setting them as cropland (value=1)
            output_raster = os.path.join(cropland_30m_dir, f'{year}_2km_{bound_no}.tif')
            filter_raster_on_threshold(input_raster=resampled_raster, output_raster=output_raster,
                                       threshold_value1=0.15,
                                       assign_value=1, refraster=resampled_raster)

        # In this step, we are mosaicin each year's datasets to form a single cropland dataset for each year
        year_list = [2008, 2009, 2010, 2011, 2015]

        for year in year_list:
            search_by = f'*{year}_2km_bound*.tif'
            raster_name = f'USDA_cropland_{year}.tif'
            mosaic_rasters(input_dir=cropland_30m_dir, output_dir=westus_processed_cropland_dir,
                           raster_name=raster_name, ref_raster=ref_raster, search_by=search_by,
                           dtype=np.float32, mosaicing_method='max', resolution=resolution, nodata=no_data_value)
            print(f'cdl data for {year} processed to cropland')

        # 2000 and 2005 cdl data isn't available for all Western US.
        # 'try_maximum_occurrence_approach' compares each pixel between 2008-2011.
        # Assigns cropland (value 1) for a pixel if that pixel is cropland in most years between 2008-2011
        if try_maximum_occurrence_approach:
            # Reading 2008-2011 cropland data in memory to apply maximum occurence approach
            westus_cdl_crops = glob(os.path.join(westus_processed_cropland_dir, '*.tif'))

            for crop_cdl in westus_cdl_crops:
                if '2008' in crop_cdl:
                    cdl_2008_arr, cdl_file = read_raster_arr_object(crop_cdl)
                    shape = cdl_2008_arr.shape
                    cdl_2008_arr = cdl_2008_arr.flatten()
                elif '2009' in crop_cdl:
                    cdl_2009_arr = read_raster_arr_object(crop_cdl, get_file=False).flatten()
                elif '2010' in crop_cdl:
                    cdl_2010_arr = read_raster_arr_object(crop_cdl, get_file=False).flatten()
                elif '2011' in crop_cdl:
                    cdl_2011_arr = read_raster_arr_object(crop_cdl, get_file=False).flatten()

            # stacking 2008-2011 cropland cdl arrays
            max_arr = np.stack([cdl_2008_arr, cdl_2009_arr, cdl_2010_arr, cdl_2011_arr], axis=0)
            # creating an empty array to assign cropland (1) if most years between 2008-2011 are cropland
            new_arr = np.zeros(cdl_2008_arr.shape)

            # this loop takes each column of stacked array. np.bincount() counts the number of occurence of each value
            # in the selected column. Then, np.argmax() selects the value which occurs most frequently. Then, we paste
            # the selected value to a new array
            for i in range(max_arr.shape[1]):
                selected_arr = max_arr[:, i]
                selected_arr = np.where(np.isnan(selected_arr), 0, selected_arr)  # replacing Nan values with 0

                max_occurred_value = np.argmax(np.bincount(list(selected_arr)))  # # # improve this. try to avoid loop
                new_arr[i] = max_occurred_value

            max_arr = new_arr.reshape(shape)

            # replacing the nan pixels with -9999 value
            ref_arr = read_raster_arr_object(ref_raster, get_file=False)
            max_arr = np.where(~np.isnan(ref_arr), max_arr, ref_arr)

        else:  # if try_maximum_occurrence_approach=False, assigns 2008 cropland arrays
            # value as 2000 and 2005 cropland data
            cdl_2008_data = glob(os.path.join(westus_processed_cropland_dir, '*2008.tif'))
            cdl_2008_arr, cdl_file = read_raster_arr_object(cdl_2008_data)
            max_arr = cdl_2008_arr.reshape(cdl_2008_arr.shape)

        # writing generated cropland rasters for 2000 and 2005
        output_2000_cdl_data = os.path.join(westus_processed_cropland_dir, 'USDA_cropland_2000.tif')
        output_2005_cdl_data = os.path.join(westus_processed_cropland_dir, 'USDA_cropland_2005.tif')

        write_array_to_raster(max_arr, raster_file=cdl_file, transform=cdl_file.transform,
                              output_path=output_2000_cdl_data)
        write_array_to_raster(max_arr, raster_file=cdl_file, transform=cdl_file.transform,
                              output_path=output_2005_cdl_data)
        print('cdl cropland data for 2000 and 2005 generated')

        # Copying processed cdl cropland data to compiled data folder and saving in a dictionary to load later
        westus_cdl_crops = glob(os.path.join(westus_processed_cropland_dir, '*.tif'))
        cdl_cropland_processing_dict = {}
        for crop_cdl in westus_cdl_crops:
            data_name = os.path.basename(crop_cdl)
            copied_data = copy_file(crop_cdl, copy_dir='../../Data_main/Compiled_data')
            cdl_cropland_processing_dict[data_name] = copied_data
            pickle.dump(cdl_cropland_processing_dict, open('../../Data_main/Compiled_data/cdl_cropland_dict.pkl',
                                                           mode='wb+'))
    else:
        cdl_cropland_processing_dict = pickle.load(open('../../Data_main/Compiled_data/cdl_cropland_dict.pkl',
                                                        mode='rb'))

    return cdl_cropland_processing_dict


def process_irrigated_landuse_data(input_dir='../../Data_main/Raster_data/Irrigated_agriculture',
                                   input_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp',
                                   skip_processing=False,
                                   ref_raster=WestUS_raster, resampling_resolution=model_res,
                                   try_maximum_occurrence_approach=True):
    """
    Process irrigated cropland data from Xie et al. 2021. Two sets of rasters are generated. First: Irrigated area
    raster with values 0 and 1 (1 menas irrigated); second: Irrigated fraction data (values in fraction from 0 or 1,
    higher fraction means higher irrigated lands in that 2-km pixel).

    :param input_dir: Filepath of input data directory.
    :param input_shape: Filepath of ROI shapefile.
    :param skip_processing: Bool. Set to True is want to skip processing.
    :param ref_raster: Filepath of Western US reference raster.
    :param resampling_resolution: Model resolution used for resampling dataset.
                                  Here, the model resolution is only used for resampling CONUS-scale dataset.
                                  At later step of masking/clipping, output raster dimension was used from the
                                  reference raster.
    :param try_maximum_occurrence_approach: Defaults to False to set 2017 dataset value as 2018-2022 datasets.
                                            'try_maximum_occurrence_approach': (set to true to use)
                                            2018-2022 irrigated cropland data aren't available.
                                            'try_maximum_occurrence_approach' compares each pixel between 2013-2017.
                                            Assigns irrigated cropland (value 1) for a pixel if that pixel is irrigated
                                            in most years between 2008-2011.

    :return: A dictionary of processed dataset filepaths. The dictionary has years (e.g., 2015) as keys.
    """
    irrigated_land_dict = None

    if not skip_processing:
        print('Processing Irrigated Agriculture dataset....')
        input_raster = glob(os.path.join(input_dir, '*.tif'))

        irrigation_land_dict = {}  # a dictionary to store irrigated area data (values in 0 or 1, 1 means irrigated)

        # creating output directory
        interim_dir = os.path.join(input_dir, 'interim_data')
        irrigated_frac_dir = os.path.join(input_dir, 'WestUS_Processed_irrigated_frac')
        irrigated_land_dir = os.path.join(input_dir, 'WestUS_Processed_irrigated_lands')
        makedirs([interim_dir, irrigated_frac_dir, irrigated_land_dir])

        for raster in input_raster:
            if any([i in raster for i in ['clipped', 'resample', 'irrigated']]):
                os.remove(raster)  # to remove raster previously processed
            else:
                # The dataset is 30m resolution CONUS scale and huge to clip and resample at the same time
                # Resampling to model resolution first and then clipping. "average" resampling calculates the
                # fraction of irrigated area (30m pixel) in each 2-km pixel.

                # # Processing irrigated area data to fraction of irrigated area data
                # resampling (by weighted average method) CONUS-scale dataset to 2km
                # Using model resolution for resampling
                resampled_raster = clip_resample_reproject_raster(
                    input_raster=raster, input_shape=input_shape, keyword='resampled', resolution=resampling_resolution,
                    output_raster_dir=interim_dir, clip_and_resample=False, clip=False, resample=True,
                    resample_algorithm='average', use_ref_width_height=False)

                # Masking resampled dataset to Western US scale
                # Using width height of reference raster. 'resolution' set to None.
                year = os.path.splitext(os.path.basename(resampled_raster))[0][-4:]
                output_name = f'irrigated_agri_frac_{year}'
                clipped_raster = clip_resample_reproject_raster(
                    input_raster=resampled_raster, input_shape=input_shape, keyword=' ', raster_name=output_name,
                    resolution=None, output_raster_dir=irrigated_frac_dir,
                    clip_and_resample=True, clip=False, resample=False, targetaligned=False,
                    use_ref_width_height=True,
                    ref_raster=ref_raster)  # targetAlignedPixels=False to ensure equal row-columns like other dataset

                # # Processing irrigated area data to presence of irrigated area data
                # Using threshold to choose irrigated area
                output_name = f'irrigated_agri_{year}'
                output_fp = os.path.join(irrigated_land_dir, f'{output_name}.tif')
                filter_raster_on_threshold(input_raster=clipped_raster, output_raster=output_fp, threshold_value1=0.15,
                                           assign_value=1)

                # copying to compiled directory
                copied_data = copy_file(input_dir_file=output_fp, copy_dir='../../Data_main/Compiled_data', rename=None)
                irrigation_land_dict[year] = copied_data

        # 2018-2022 irrigated cropland data aren't available.
        # 'try_maximum_occurrence_approach' compares each pixel between 2013-2017.
        # Assigns irrigated cropland (value 1) for a pixel if that pixel is irrigated
        # in most years between 2008-2011
        irrig_2013_arr, irrig_2014_arr, irrig_2015_arr, irrig_2016_arr, irrig_2017_arr = None, None, None, \
                                                                                         None, None
        if try_maximum_occurrence_approach:
            # Reading 2013-2017 cropland data in memory to apply maximum occurrence approach
            westus_irrigated_crops = glob(os.path.join(irrigated_land_dir, '*201[3-7].tif'))
            print(westus_irrigated_crops)

            for irrigated_crop in westus_irrigated_crops:
                if '2013' in irrigated_crop:
                    irrig_2013_arr, irrig_file = read_raster_arr_object(irrigated_crop)
                    shape = irrig_2013_arr.shape
                    irrig_2013_arr = irrig_2013_arr.flatten()
                elif '2014' in irrigated_crop:
                    irrig_2014_arr = read_raster_arr_object(irrigated_crop, get_file=False).flatten()
                elif '2015' in irrigated_crop:
                    irrig_2015_arr = read_raster_arr_object(irrigated_crop, get_file=False).flatten()
                elif '2016' in irrigated_crop:
                    irrig_2016_arr = read_raster_arr_object(irrigated_crop, get_file=False).flatten()
                elif '2017' in irrigated_crop:
                    irrig_2017_arr = read_raster_arr_object(irrigated_crop, get_file=False).flatten()

            # stacking 2013-2017 irrigated cropland arrays
            max_arr = np.stack([irrig_2013_arr, irrig_2014_arr, irrig_2015_arr, irrig_2016_arr, irrig_2017_arr],
                               axis=0)
            # creating an empty array to assign cropland (1) if most years between 2008-2011 are cropland
            new_arr = np.zeros(irrig_2017_arr.shape)

            # this loop takes each column of stacked array. np.bincount() counts the number of occurence of
            # each value in the selected column. Then, np.argmax() selects the value which occurs most
            # frequently. Then, we paste the selected value to a new array
            for i in range(max_arr.shape[1]):
                selected_arr = max_arr[:, i]
                selected_arr = np.where(np.isnan(selected_arr), 0, selected_arr)  # replacing Nan values with 0

                max_occurred_value = np.argmax(
                    np.bincount(list(selected_arr)))
                new_arr[i] = max_occurred_value

            max_arr = new_arr.reshape(shape)

            # replacing the nan pixels with -9999 value
            ref_arr = read_raster_arr_object(ref_raster, get_file=False)
            max_arr = np.where(~np.isnan(ref_arr), max_arr, ref_arr)

        else:  # if try_maximum_occurrence_approach=False, assigns 2017 irrigated cropland arrays value
            # as 2018-2022 irrigated cropland data
            irrig_crop_2017_data = glob(os.path.join(irrigated_land_dir, '*2017.tif'))
            irrig_2017_arr, irrig_file = read_raster_arr_object(irrig_crop_2017_data)
            max_arr = irrig_2017_arr.reshape(irrig_2017_arr.shape)

        # writing generated cropland rasters for 2018-2022
        output_2018_irrigated_data = os.path.join(irrigated_land_dir, 'irrigated_agri_2018.tif')
        output_2019_irrigated_data = os.path.join(irrigated_land_dir, 'irrigated_agri_2019.tif')
        output_2020_irrigated_data = os.path.join(irrigated_land_dir, 'irrigated_agri_2020.tif')
        output_2021_irrigated_data = os.path.join(irrigated_land_dir, 'irrigated_agri_2021.tif')
        output_2022_irrigated_data = os.path.join(irrigated_land_dir, 'irrigated_agri_2022.tif')

        write_copy_files = [output_2018_irrigated_data, output_2019_irrigated_data, output_2020_irrigated_data,
                            output_2021_irrigated_data, output_2022_irrigated_data]

        for each_data in write_copy_files:
            # writing created file
            write_array_to_raster(max_arr, raster_file=irrig_file, transform=irrig_file.transform,
                                  output_path=each_data)
            # copying to compiled directory
            copied_data = copy_file(input_dir_file=each_data, copy_dir='../../Data_main/Compiled_data', rename=None)
            year = os.path.splitext(os.path.basename(each_data))[0][-4:]
            irrigation_land_dict[year] = copied_data

            print('Irrigated cropland data for 2018-2022 generated')

        print('Processed Irrigated Agriculture dataset')
        pickle.dump(irrigation_land_dict, open('../../Data_main/Compiled_data/irrigated_land_dict.pkl', mode='wb+'))

    else:
        irrigated_land_dict = pickle.load(open('../../Data_main/Compiled_data/irrigated_land_dict.pkl', mode='rb'))

    return irrigated_land_dict


def process_ssebop_data(input_ssebop_dir, output_dir_ssebop_monthly,output_dir_ssebop_yearly,
                        years=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013,
                               2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022),
                        west_US_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp',
                        ref_raster=WestUS_raster, resolution=model_res, skip_processing=False):
    """
    Process (sum and clip to Western US extent) Ssebop ET.

    :param input_ssebop_dir: Directory file path of downloaded ssebop datasets.
    :param output_dir_ssebop_monthly: File path of directory to save monthly SseBop ET at Western US extent.
    :param output_dir_ssebop_yearly: File path of directory to save summed ET for each year at Western US extent.
    :param years: Tuple/list of years for which ssebop data was downloaded.
    :param west_US_shape: Filepath of Western US shapefile.
    :param ref_raster: Model reference raster filepath.
    :param resolution: Resolution used in the model. Default set to model_res = 0.02000000000000000736.
    :param skip_processing: Set to True if want to skip ssebop processing.

    :return: A dictionary with ssebop data file paths.
    """
    if not skip_processing:
        interim_dir_for_clip = os.path.join(input_ssebop_dir, 'interim_dir_clip')
        interim_dir_for_sum = os.path.join(input_ssebop_dir, 'interim_dir_sum')
        makedirs([interim_dir_for_clip, interim_dir_for_sum, output_dir_ssebop_monthly, output_dir_ssebop_yearly])

        #########
        # # Code-block for saving monthly data for the Western US
        #########
        # Clipping SseBop monthly datasets for Western US
        monthly_ssebop_data = glob(os.path.join(input_ssebop_dir, '*.tif'))  # monthly ssebop datasets
        for data in monthly_ssebop_data:
            month = os.path.basename(data).split('.')[0][-2:]
            year = os.path.basename(data).split('.')[0][1:5]

            monthly_raster_name = f'SSEBOP_ET_{year}_{month}.tif'

            clip_resample_reproject_raster(input_raster=data,
                                           input_shape=west_US_shape,
                                           raster_name=monthly_raster_name, keyword=' ',
                                           output_raster_dir=output_dir_ssebop_monthly,
                                           clip=False, resample=False, clip_and_resample=True,
                                           targetaligned=True, resample_algorithm='near',
                                           use_ref_width_height=True, ref_raster=ref_raster,
                                           resolution=resolution)

        #########
        # # Code-block for summing monthly data for years by growing season for the Western US
        #########
        ssebop_processing_dict = {}

        # Growing season shapefile
        grow_season_Apr_Oct_shp = '../../Data_main/shapefiles/Growing_season_zones/growing_season_Apr_Oct.shp'
        grow_season_Jan_Dec_shp = '../../Data_main/shapefiles/Growing_season_zones/growing_season_Jan_Dec.shp'
        grow_season_Jul_Aug_shp = '../../Data_main/shapefiles/Growing_season_zones/growing_season_Jul_Aug.shp'
        grow_season_Mar_Nov_shp = '../../Data_main/shapefiles/Growing_season_zones/growing_season_Mar_Nov.shp'
        grow_season_May_Sep_shp = '../../Data_main/shapefiles/Growing_season_zones/growing_season_May_Sep.shp'

        # Assigning growing season shapefiles in a dictionary
        grow_season_shape_dict = {'Apr_Oct': grow_season_Apr_Oct_shp, 'Jan_Dec': grow_season_Jan_Dec_shp,
                                  'Jul_Aug': grow_season_Jul_Aug_shp, 'Mar_Nov': grow_season_Mar_Nov_shp,
                                  'May_sep': grow_season_May_Sep_shp}

        # Creating a dictionary of growing seasons' month span
        month_dict = {'Apr_Oct': range(4, 11), 'Jan_Dec': range(1, 13), 'Jul_Aug': range(7, 9),
                      'Mar_Nov': range(3, 12), 'May_sep': range(5, 10)}

        for year in years:  # first loop for years
            print(f'Processing Ssebop data for {year}...')
            ssebop_datasets = glob(os.path.join(input_ssebop_dir, f'*{year}*.tif'))  # monthly ssebop datasets for each year

            for season in grow_season_shape_dict.keys():  # second loop for growing season zones
                growing_season_shape = grow_season_shape_dict[season]
                months = list(month_dict[season])

                ssebop_clipped_list = []
                for monthly_ssebop in ssebop_datasets:  # third loop for selected monthly ssebop data for each year
                    ssebop_month = os.path.basename(monthly_ssebop).split('.')[0][-2:]  # month extracted as string
                    ssebop_month = int(ssebop_month)  # month converted to integer

                    if ssebop_month in months:
                        clipped_ssebop = \
                            clip_resample_reproject_raster(input_raster=monthly_ssebop,
                                                           input_shape=growing_season_shape,
                                                           keyword=' ', output_raster_dir=interim_dir_for_clip,
                                                           clip=False, resample=False, clip_and_resample=True,
                                                           targetaligned=True, resample_algorithm='near',
                                                           resolution=resolution)

                        ssebop_clipped_list.append(clipped_ssebop)

                # Summing raster for each growing season
                summed_output_for_season = os.path.join(interim_dir_for_sum, f'ssebop_{year}_' + season + '.tif')
                sum_rasters(raster_list=ssebop_clipped_list, raster_dir=None, output_raster=summed_output_for_season,
                            ref_raster=ssebop_clipped_list[0])

            # mosaicing ssebop ET for all growing seasons in a year
            yearly_summed_ssebop = f'SSEBOP_ET_{year}.tif'
            search_by = f'*{year}*.tif'
            _, mosaiced_ssebop = mosaic_rasters(input_dir=interim_dir_for_sum, output_dir=output_dir_ssebop_yearly,
                                                raster_name=yearly_summed_ssebop, ref_raster=ref_raster,
                                                search_by=search_by, resolution=resolution)

            # Copying mosaiced ssebop ET to compiled data folder and saving in a dictionary to load later
            copied_data = copy_file(mosaiced_ssebop, copy_dir='../../Data_main/Compiled_data')
            ssebop_processing_dict[yearly_summed_ssebop[:yearly_summed_ssebop.rfind('.')]] = copied_data
            pickle.dump(ssebop_processing_dict, open('../../Data_main/Compiled_data/ssebop_dict.pkl', mode='wb+'))
    else:
        ssebop_processing_dict = pickle.load(open('../../Data_main/Compiled_data/ssebop_dict.pkl', mode='rb'))

        return ssebop_processing_dict


def convert_prism_data_to_tif(input_dir, output_dir, keyword='prism_precip'):
    """
    Convert prism rainfall/temperature datasets from .bil format to GeoTiff format.

    Download PRISM datasets directly from  'https://prism.oregonstate.edu/recent/'

    :param input_dir: Directory path of prism data in .bil format.
    :param output_dir: Directory path of converted (.tif) prism data.
    :param keyword: keyword to add before processed datasets.

    :return: None.
    """
    makedirs([output_dir])
    prism_datasets = glob(os.path.join(input_dir, '*.bil'))

    for data in prism_datasets:
        year_month = os.path.basename(data).split('_')[-2]
        output_name = keyword + '_' + year_month + '.tif'
        output_file = os.path.join(output_dir, output_name)
        gdal.Translate(destName=output_file, srcDS=data, format='GTiff', outputType=gdal.GDT_Float32,
                       outputSRS='EPSG:4269')


def process_prism_data(prism_bil_dir, prism_tif_dir, output_dir_prism_monthly, output_dir_prism_yearly,
                       years=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012,
                              2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022),
                       keyword='prism_precip',
                       west_US_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp',
                       ref_raster=WestUS_raster, resolution=model_res, skip_processing=False):
    """
    Process (sum and clip to Western US extent) Prism Precipitation.

    :param prism_bil_dir: Directory file path of downloaded prism datasets in .bil format.
    :param prism_tif_dir: Directory file path of prism datasets converted to tif format.
    :param output_dir_prism_monthly: File path of directory to save monthly prism precipitation/temperature data for
                                     at Western US extent.
    :param output_dir_prism_yearly: File path of directory to save summed/mean prism precipitation/temperature data for
                                    each year at Western US extent.
    :param years: Tuple/list of years for which prism data was downloaded.
    :param keyword: keyword to add before processed datasets.
    :param west_US_shape: Filepath of Western US shapefile.
    :param ref_raster: Model reference raster filepath.
    :param resolution: Resolution used in the model. Default set to model_res = 0.02000000000000000736.
    :param skip_processing: Set to True if want to skip prism precip processing.

    :return: A dictionary with prism precip data file paths.
    """
    if not skip_processing:
        convert_prism_data_to_tif(input_dir=prism_bil_dir, output_dir=prism_tif_dir, keyword=keyword)

        interim_dir_for_clip = os.path.join(prism_tif_dir, 'interim_dir_clip')

        # Creating additional folders/directories needed for processing (clip/sum/mean)
        if 'precip' in keyword:
            interim_dir_for_sum = os.path.join(prism_tif_dir, 'interim_dir_sum')
            makedirs([interim_dir_for_clip, interim_dir_for_sum, output_dir_prism_yearly])
        elif 'temp' in keyword:
            interim_dir_for_mean = os.path.join(prism_tif_dir, 'interim_dir_mean')
            makedirs([interim_dir_for_clip, interim_dir_for_mean, output_dir_prism_monthly, output_dir_prism_yearly])


        #########
        # # Code-block for saving monthly data for the Western US
        #########
        # Clipping Prism monthly datasets for Western US
        monthly_prism_tifs = glob(os.path.join(prism_tif_dir, '*.tif'))  # monthly prism datasets
        for data in monthly_prism_tifs:
            month = os.path.basename(data).split('.')[0][-2:]
            year = os.path.basename(data).split('.')[0].split('_')[2][:4]
            if 'precip' in keyword:
                monthly_raster_name = f'prism_precip_{year}_{month}.tif'
            else:
                monthly_raster_name = f'prism_temp_{year}_{month}.tif'

            clip_resample_reproject_raster(input_raster=data,
                                           input_shape=west_US_shape,
                                           raster_name=monthly_raster_name, keyword=' ',
                                           output_raster_dir=output_dir_prism_monthly,
                                           clip=False, resample=False, clip_and_resample=True,
                                           targetaligned=True, resample_algorithm='near',
                                           use_ref_width_height=True, ref_raster=ref_raster,
                                           resolution=resolution)

        #########
        # # Code-block for summing monthly data for years by growing season for the Western US
        #########
        prism_processing_dict = {}

        # Growing season shapefile
        grow_season_Apr_Oct_shp = '../../Data_main/shapefiles/Growing_season_zones/growing_season_Apr_Oct.shp'
        grow_season_Jan_Dec_shp = '../../Data_main/shapefiles/Growing_season_zones/growing_season_Jan_Dec.shp'
        grow_season_Jul_Aug_shp = '../../Data_main/shapefiles/Growing_season_zones/growing_season_Jul_Aug.shp'
        grow_season_Mar_Nov_shp = '../../Data_main/shapefiles/Growing_season_zones/growing_season_Mar_Nov.shp'
        grow_season_May_Sep_shp = '../../Data_main/shapefiles/Growing_season_zones/growing_season_May_Sep.shp'

        # Assigning growing season shapefiles in a dictionary
        grow_season_shape_dict = {'Apr_Oct': grow_season_Apr_Oct_shp, 'Jan_Dec': grow_season_Jan_Dec_shp,
                                  'Jul_Aug': grow_season_Jul_Aug_shp, 'Mar_Nov': grow_season_Mar_Nov_shp,
                                  'May_sep': grow_season_May_Sep_shp}

        # Creating a dictionary of growing seasons' month span
        month_dict = {'Apr_Oct': range(4, 11), 'Jan_Dec': range(1, 13), 'Jul_Aug': range(7, 9),
                      'Mar_Nov': range(3, 12), 'May_sep': range(5, 10)}

        for year in years:  # first loop for years
            print(f'Processing {keyword} data for {year}...')
            prism_datasets = glob(os.path.join(prism_tif_dir, f'*{year}*.tif'))  # monthly prism datasets for each year

            for season in grow_season_shape_dict.keys():  # second loop for growing season zones
                growing_season_shape = grow_season_shape_dict[season]
                months = list(month_dict[season])

                prism_clipped_list = []
                for monthly_prism in prism_datasets:  # third loop for selected monthly prism data for each year
                    prism_month = os.path.basename(monthly_prism).split('.')[0][-2:]  # month extracted as string
                    prism_month = int(prism_month)  # month converted to integer

                    if prism_month in months:
                        clipped_prism = \
                            clip_resample_reproject_raster(input_raster=monthly_prism,
                                                           input_shape=growing_season_shape,
                                                           keyword=' ', output_raster_dir=interim_dir_for_clip,
                                                           clip=False, resample=False, clip_and_resample=True,
                                                           targetaligned=True, resample_algorithm='near',
                                                           resolution=resolution)

                        prism_clipped_list.append(clipped_prism)

                if 'precip' in keyword:
                    # Summing raster for each growing season
                    summed_output_for_season = os.path.join(interim_dir_for_sum, f'prism_{year}_' + season + '.tif')
                    sum_rasters(raster_list=prism_clipped_list, raster_dir=None, output_raster=summed_output_for_season,
                                ref_raster=prism_clipped_list[0])

                    # mosaicing prism precip for all growing seasons in a year
                    yearly_prism = f'{keyword}_{year}.tif'
                    search_by = f'*{year}*.tif'
                    _, mosaiced_prism = mosaic_rasters(input_dir=interim_dir_for_sum,
                                                       output_dir=output_dir_prism_yearly,
                                                       raster_name=yearly_prism, ref_raster=ref_raster,
                                                       search_by=search_by, resolution=resolution)

                elif 'temp' in keyword:
                    # Calculating mean of raster for each growing season
                    mean_output_for_season = os.path.join(interim_dir_for_mean, f'prism_{year}_' + season + '.tif')
                    mean_rasters(raster_list=prism_clipped_list, raster_dir=None, output_raster=mean_output_for_season,
                                 ref_raster=prism_clipped_list[0])

                    # mosaicing prism temperature for all growing seasons in a year
                    yearly_prism = f'{keyword}_{year}.tif'
                    search_by = f'*{year}*.tif'
                    _, mosaiced_prism = mosaic_rasters(input_dir=interim_dir_for_mean,
                                                       output_dir=output_dir_prism_yearly,
                                                       raster_name=yearly_prism, ref_raster=ref_raster,
                                                       search_by=search_by, resolution=resolution)

            # Copying mosaiced prism precip to compiled data folder and saving in a dictionary to load later
            copied_data = copy_file(mosaiced_prism, copy_dir='../../Data_main/Compiled_data')
            prism_processing_dict[yearly_prism[:yearly_prism.rfind('.')]] = copied_data
            pickle.dump(prism_processing_dict, open('../../Data_main/Compiled_data/prism_dict.pkl', mode='wb+'))
    else:
        prism_processing_dict = pickle.load(open('../../Data_main/Compiled_data/prism_dict.pkl', mode='rb'))

        return prism_processing_dict


def process_terracimate_et_data(years=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012,
                                       2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022),
                                terraclimate_et_dir='../../Data_main/Raster_data/TERRACLIMATE_ET/WestUS_monthly',
                                output_dir_terraclimate='../../Data_main/Raster_data/TERRACLIMATE_ET/WestUS',
                                ref_raster=WestUS_raster, resolution=model_res, skip_processing=False):
    """
    Process (sum and clip to Western US extent) TerraClimate ET.

    :param years: Tuple/list of years for which TerraClimate data was downloaded.
    :param terraclimate_et_dir: Directory file path of downloaded terraclimate ET datasets.
    :param output_dir_terraclimate: File path of directory to save summed ET for each year at Western US extent.
    :param ref_raster: Model reference raster filepath.
    :param resolution: Resolution used in the model. Default set to model_res = 0.02000000000000000736.
    :param skip_processing: Set to True if want to skip terraclimate ET processing.

    :return: A dictionary with Terraclimate ET data file paths.
    """
    if not skip_processing:
        interim_dir_for_clip = os.path.join(terraclimate_et_dir, 'interim_dir_clip')
        interim_dir_for_sum = os.path.join(terraclimate_et_dir, 'interim_dir_sum')
        makedirs([interim_dir_for_clip, interim_dir_for_sum, output_dir_terraclimate])

        terraclimate_processing_dict = {}

        # Growing season shapefile
        grow_season_Apr_Oct_shp = '../../Data_main/shapefiles/Growing_season_zones/growing_season_Apr_Oct.shp'
        grow_season_Jan_Dec_shp = '../../Data_main/shapefiles/Growing_season_zones/growing_season_Jan_Dec.shp'
        grow_season_Jul_Aug_shp = '../../Data_main/shapefiles/Growing_season_zones/growing_season_Jul_Aug.shp'
        grow_season_Mar_Nov_shp = '../../Data_main/shapefiles/Growing_season_zones/growing_season_Mar_Nov.shp'
        grow_season_May_Sep_shp = '../../Data_main/shapefiles/Growing_season_zones/growing_season_May_Sep.shp'

        # Assigning growing season shapefiles in a dictionary
        grow_season_shape_dict = {'Apr_Oct': grow_season_Apr_Oct_shp, 'Jan_Dec': grow_season_Jan_Dec_shp,
                                  'Jul_Aug': grow_season_Jul_Aug_shp, 'Mar_Nov': grow_season_Mar_Nov_shp,
                                  'May_sep': grow_season_May_Sep_shp}

        # Creating a dictionary of growing seasons' month span
        month_dict = {'Apr_Oct': range(4, 11), 'Jan_Dec': range(1, 13), 'Jul_Aug': range(7, 9),
                      'Mar_Nov': range(3, 12), 'May_sep': range(5, 10)}

        for year in years:  # first loop for years
            print(f'Processing TerraClimate ET data for {year}...')
            terraclimate_et_datasets = glob(
                os.path.join(terraclimate_et_dir, f'*{year}*.tif'))  # monthly ET datasets for each year

            for season in grow_season_shape_dict.keys():  # second loop for growing season zones
                growing_season_shape = grow_season_shape_dict[season]
                months = list(month_dict[season])

                terraclimate_clipped_list = []
                for monthly_et in terraclimate_et_datasets:  # third loop for selected monthly terraclimate data for each year
                    terraclimate_month = os.path.basename(monthly_et).split('.')[0].split('_')[
                        -1]  # month extracted as string
                    terraclimate_month = int(terraclimate_month)  # month converted to integer

                    if terraclimate_month in months:
                        clipped_et = \
                            clip_resample_reproject_raster(input_raster=monthly_et,
                                                           input_shape=growing_season_shape,
                                                           keyword=' ', output_raster_dir=interim_dir_for_clip,
                                                           clip=False, resample=False, clip_and_resample=True,
                                                           targetaligned=True, resample_algorithm='near',
                                                           resolution=resolution)

                        terraclimate_clipped_list.append(clipped_et)

                # Summing raster for each growing season
                summed_output_for_season = os.path.join(interim_dir_for_sum, f'terraclimate_{year}_' + season + '.tif')
                sum_rasters(raster_list=terraclimate_clipped_list, raster_dir=None,
                            output_raster=summed_output_for_season, ref_raster=terraclimate_clipped_list[0])

            # mosaicing TerraClimate ET for all growing seasons in a year
            yearly_summed_terraclimate = f'TERRACLIMATE_ET_{year}.tif'
            search_by = f'*{year}*.tif'
            _, mosaiced_terraclimate = mosaic_rasters(input_dir=interim_dir_for_sum, output_dir=output_dir_terraclimate,
                                                      raster_name=yearly_summed_terraclimate, ref_raster=ref_raster,
                                                      search_by=search_by, resolution=resolution)

            # Copying mosaiced terraclimate ET to compiled data folder and saving in a dictionary to load later
            copied_data = copy_file(mosaiced_terraclimate, copy_dir='../../Data_main/Compiled_data')
            terraclimate_processing_dict[
                yearly_summed_terraclimate[:yearly_summed_terraclimate.rfind('.')]] = copied_data
            pickle.dump(terraclimate_processing_dict, open('../../Data_main/Compiled_data/terraclimate_dict.pkl',
                                                           mode='wb+'))
    else:
        terraclimate_processing_dict = pickle.load(open('../../Data_main/Compiled_data/terraclimate_dict.pkl',
                                                        mode='rb'))

        return terraclimate_processing_dict


def process_OpenET_data(years=(2016, 2017, 2018, 2019, 2020, 2021, 2022),
                        OpenET_monthly_dir='../../Data_main/Raster_data/OpenET_ensemble/WestUS_monthly',
                        output_dir_OpenET='../../Data_main/Raster_data/OpenET_ensemble/WestUS',
                        ref_raster=WestUS_raster, resolution=model_res, skip_processing=False):
    """
    Process (sum and clip to Western US extent) TerraClimate ET.

    :param years: Tuple/list of years for which TerraClimate data was downloaded.
    :param OpenET_monthly_dir: Directory file path of downloaded OpenET monthly datasets.
    :param output_dir_OpenET: File path of directory to save summed ET for each year at Western US extent.
    :param ref_raster: Model reference raster filepath.
    :param resolution: Resolution used in the model. Default set to model_res = 0.02000000000000000736.
    :param skip_processing: Set to True if want to skip terraclimate ET processing.

    :return: A dictionary with OpenET data file paths.
    """
    if not skip_processing:
        interim_dir_for_clip = os.path.join(OpenET_monthly_dir, 'interim_dir_clip')
        interim_dir_for_sum = os.path.join(OpenET_monthly_dir, 'interim_dir_sum')
        makedirs([interim_dir_for_clip, interim_dir_for_sum, output_dir_OpenET])

        OpenET_processing_dict = {}

        # Growing season shapefile
        grow_season_Apr_Oct_shp = '../../Data_main/shapefiles/Growing_season_zones/growing_season_Apr_Oct.shp'
        grow_season_Jan_Dec_shp = '../../Data_main/shapefiles/Growing_season_zones/growing_season_Jan_Dec.shp'
        grow_season_Jul_Aug_shp = '../../Data_main/shapefiles/Growing_season_zones/growing_season_Jul_Aug.shp'
        grow_season_Mar_Nov_shp = '../../Data_main/shapefiles/Growing_season_zones/growing_season_Mar_Nov.shp'
        grow_season_May_Sep_shp = '../../Data_main/shapefiles/Growing_season_zones/growing_season_May_Sep.shp'

        # Assigning growing season shapefiles in a dictionary
        grow_season_shape_dict = {'Apr_Oct': grow_season_Apr_Oct_shp, 'Jan_Dec': grow_season_Jan_Dec_shp,
                                  'Jul_Aug': grow_season_Jul_Aug_shp, 'Mar_Nov': grow_season_Mar_Nov_shp,
                                  'May_sep': grow_season_May_Sep_shp}

        # Creating a dictionary of growing seasons' month span
        month_dict = {'Apr_Oct': range(4, 11), 'Jan_Dec': range(1, 13), 'Jul_Aug': range(7, 9),
                      'Mar_Nov': range(3, 12), 'May_sep': range(5, 10)}

        for year in years:  # first loop for years
            print(f'Processing OpenET data for {year}...')
            OpenET_datasets = glob(
                os.path.join(OpenET_monthly_dir, f'*{year}*.tif'))  # monthly ET datasets for each year

            for season in grow_season_shape_dict.keys():  # second loop for growing season zones
                growing_season_shape = grow_season_shape_dict[season]
                months = list(month_dict[season])

                OpenET_clipped_list = []
                for monthly_et in OpenET_datasets:  # third loop for selected monthly terraclimate data for each year
                    OpenET_month = os.path.basename(monthly_et).split('.')[0].split('_')[
                        -1]  # month extracted as string
                    OpenET_month = int(OpenET_month)  # month converted to integer

                    if OpenET_month in months:
                        clipped_et = \
                            clip_resample_reproject_raster(input_raster=monthly_et,
                                                           input_shape=growing_season_shape,
                                                           keyword=' ', output_raster_dir=interim_dir_for_clip,
                                                           clip=False, resample=False, clip_and_resample=True,
                                                           targetaligned=True, resample_algorithm='near',
                                                           resolution=resolution)

                        OpenET_clipped_list.append(clipped_et)

                # Summing raster for each growing season
                summed_output_for_season = os.path.join(interim_dir_for_sum, f'OpenET_{year}_' + season + '.tif')
                sum_rasters(raster_list=OpenET_clipped_list, raster_dir=None,
                            output_raster=summed_output_for_season, ref_raster=OpenET_clipped_list[0])

            # mosaicing OpenET for all growing seasons in a year
            yearly_summed_OpenET = f'OpenET_{year}.tif'
            search_by = f'*{year}*.tif'
            _, mosaiced_terraclimate = mosaic_rasters(input_dir=interim_dir_for_sum, output_dir=output_dir_OpenET,
                                                      raster_name=yearly_summed_OpenET, ref_raster=ref_raster,
                                                      search_by=search_by, resolution=resolution)

            # Copying mosaiced OpenET to compiled data folder and saving in a dictionary to load later
            copied_data = copy_file(mosaiced_terraclimate, copy_dir='../../Data_main/Compiled_data')
            OpenET_processing_dict[yearly_summed_OpenET[:yearly_summed_OpenET.rfind('.')]] = copied_data
            pickle.dump(OpenET_processing_dict, open('../../Data_main/Compiled_data/OpenET_dict.pkl',
                                                     mode='wb+'))
    else:
        OpenET_processing_dict = pickle.load(open('../../Data_main/Compiled_data/OpenET_dict.pkl',
                                                  mode='rb'))

        return OpenET_processing_dict


def run_all_preprocessing(skip_cdl_cropland_processing=False, cdl_already_masked=False,
                          try_maximum_occurrence_approach_for_cdl=True,
                          skip_ssebop_processing=False,
                          skip_irrigatedCrop_processing=False, skip_prism_processing=False,
                          skip_terraclimate_et_processing=False,
                          skip_OpenET_processing=False):
    """
    Run all preprocessing steps.

    :param skip_cdl_cropland_processing: Set to True to skip cdl cropland dataset processing.
    :param cdl_already_masked: Set to True to skip masking (if masked cdl data has already been prepared).
    :param try_maximum_occurrence_approach_for_cdl: Defaults to False to set 2008 dataset value as 2000 & 2005 datasets.
                                            'try_maximum_occurrence_approach': (set to True to use)
                                            2000 and 2005 cdl datasets for whole Western US isn't available.
                                            This approach sets the maximum occurring cropland/non-cropland values from
                                            2008-2011 as 2000 & 2005 datasets.
    :param skip_ssebop_processing: Set True if want to skip ssebop data preprocessing.
    :param skip_irrigatedCrop_processing: Set True if want to skip Irrigated cropland data preprocessing.
    :param skip_prism_processing: Set True if want to skip prism (precipitation and temperature) data preprocessing.
    :param skip_terraclimate_et_processing: Set True if want to skip terraclimate ET data preprocessing.
    :param skip_OpenET_processing: Set True if want to skip OpenET data preprocessing.

    :return: Dictionaries of crop and developed land use data file paths
    """
    # USDA cdl to cropland processing
    cdl_cropland_dict = \
        process_cdl_data_to_cropland(cdl_dir='../../Data_main/Raster_data/USDA_CDL/CONUS_original_30m',
                                     processing_shp_dir='../../Data_main/Raster_data/USDA_CDL/processing_shapes',
                                     westus_30m_dir='../../Data_main/Raster_data/USDA_CDL/WestUS_30m',
                                     westus_processed_cropland_dir='../../Data_main/Raster_data/USDA_CDL/WestUS_Processed_cropland',
                                     ref_raster=WestUS_raster, resolution=model_res,
                                     already_masked=cdl_already_masked,
                                     try_maximum_occurrence_approach=try_maximum_occurrence_approach_for_cdl,
                                     skip_processing=skip_cdl_cropland_processing)

    # irrigated cropland data processing
    irrigated_crop_dict = \
        process_irrigated_landuse_data(input_dir='../../Data_main/Raster_data/Irrigated_agriculture',
                                       input_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp',
                                       ref_raster=WestUS_raster, try_maximum_occurrence_approach=True,
                                       skip_processing=skip_irrigatedCrop_processing)

    # ssebop ET data processing
    ssebop_et_dict = \
        process_ssebop_data(input_ssebop_dir='../../Data_main/Raster_data/Ssebop_ETa/raw_data',
                            output_dir_ssebop_monthly='../../Data_main/Raster_data/Ssebop_ETa/WestUS_monthly',
                            output_dir_ssebop_yearly='../../Data_main/Raster_data/Ssebop_ETa/WestUS',
                            years=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012,
                                   2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022),
                            ref_raster=WestUS_raster,
                            skip_processing=skip_ssebop_processing)

    # prism precipitation data processing
    prism_precip_dict = \
        process_prism_data(years=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012,
                                  2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022),
                           prism_bil_dir='../../Data_main/Raster_data/PRISM_PRECIP/bil_format',
                           prism_tif_dir='../../Data_main/Raster_data/PRISM_PRECIP/tif_format',
                           output_dir_prism_monthly='../../Data_main/Raster_data/PRISM_PRECIP/WestUS_monthly',
                           output_dir_prism_yearly='../../Data_main/Raster_data/PRISM_PRECIP/WestUS',
                           west_US_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp',
                           keyword='prism_precip', skip_processing=skip_prism_processing)

    # prism maximum temperature data processing
    prism_temp_dict = \
        process_prism_data(years=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012,
                                  2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022),
                           prism_bil_dir='../../Data_main/Raster_data/PRISM_TEMP/bil_format',
                           prism_tif_dir='../../Data_main/Raster_data/PRISM_TEMP/tif_format',
                           output_dir_prism_monthly='../../Data_main/Raster_data/PRISM_TEMP/WestUS_monthly',
                           output_dir_prism_yearly='../../Data_main/Raster_data/PRISM_TEMP/WestUS',
                           west_US_shape='../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp',
                           keyword='prism_temp', skip_processing=skip_prism_processing)

    # terraclimate ET data processing
    terraclimate_dict = \
        process_terracimate_et_data(years=(2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012,
                                           2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022),
                                    terraclimate_et_dir='../../Data_main/Raster_data/TERRACLIMATE_ET/WestUS_monthly',
                                    output_dir_terraclimate='../../Data_main/Raster_data/TERRACLIMATE_ET/WestUS',
                                    ref_raster=WestUS_raster, resolution=model_res,
                                    skip_processing=skip_terraclimate_et_processing)

    # OpenET data processing
    OpenET_processing_dict = \
        process_OpenET_data(years=(2016, 2017, 2018, 2019, 2020, 2021, 2022),
                            OpenET_monthly_dir='../../Data_main/Raster_data/OpenET_ensemble/WestUS_monthly',
                            output_dir_OpenET='../../Data_main/Raster_data/OpenET_ensemble/WestUS',
                            ref_raster=WestUS_raster, resolution=model_res, skip_processing=skip_OpenET_processing)

    return cdl_cropland_dict, ssebop_et_dict, irrigated_crop_dict, prism_precip_dict, prism_temp_dict, \
           terraclimate_dict, OpenET_processing_dict

# def download_OpenET_data_monthly(data_name, download_dir, year_list, month_range, merge_keyword, grid_shape,
#                               refraster=WestUS_raster):
#     """
#     Download data (at monthly scale) from GEE.
#
#     :param data_name: Data name.
#     Current valid data names are -
#         ['SMAP_SM', 'LANDSAT_NDWI', 'LANDSAT_NDVI', 'GPM_PRECIP', 'GRIDMET_PRECIP', 'PRISM_PRECIP', 'VIIRS_NDVI',
#         'VIIRS_EVI', 'VIIRS_EVI2', 'MODIS_Day_LST', 'MODIS_Terra_NDVI', 'MODIS_Terra_EVI', 'MODIS_NDWI', 'MODIS_LAI',
#         'MODIS_ET', 'FAO_ACTUAL_ET', 'FAO_EVAPORATION', 'FLDAS_BF_GW_RO', 'FLDAS_SM', 'TERRSCLIMATE_ET']
#     :param download_dir: File path of download directory.
#     :param year_list: List of years to download data for.
#     :param month_range: Tuple of month ranges to download data for, e.g., for months 1-12 use (1, 12).
#     :param merge_keyword: Keyword to use for merging downloaded data. Suggested 'WestUS'/'Conus'.
#     :param grid_shape: File path of grid shape for which data will be downloaded and mosaiced.
#     :param refraster: Reference raster to use for merging downloaded datasets.
#
#     :return: File path of downloaded raster data.
#     """
#     ee.Initialize()
#     download_dir = os.path.join(download_dir, data_name)
#     makedirs([download_dir])
#
#     # Extracting dataset information required for downloading from GEE
#     data, band, multiply_scale, reducer, month_start_range, month_end_range, \
#             year_start_range, year_end_range = get_gee_dict(data_name)
#
#     # Loading grid files to be used for data download
#     grids = gpd.read_file(grid_shape)
#     grids = grids.sort_values(by='grid_no', ascending=True)
#     grid_geometry = grids['geometry']
#     grid_no = grids['grid_no']
#
#     month_list = [m for m in range(month_range[0], month_range[1] + 1)]  # creating list of months
#     for year in year_list:  # first loop for years
#         for month in month_list:  # second loop for months
#             start_date = ee.Date.fromYMD(year, month, 1)
#             start_date_dt = datetime(year, month, 1)
#
#             if month < 12:
#                 end_date = ee.Date.fromYMD(year, month + 1, 1)
#                 end_date_dt = datetime(year, month + 1, 1)
#
#             else:
#                 end_date = ee.Date.fromYMD(year + 1, 1, 1)  # for month 12 moving end date to next year
#                 end_date_dt = datetime(year + 1, 1, 1)
#
#             # a condition to check whether start and end date falls in the available data range in GEE
#             # if not the block will not be executed
#             if (start_date_dt >= month_start_range) & (end_date_dt <= month_end_range):
#
#                 for grid_sr, geometry in zip(grid_no, grid_geometry):  # third loop for grids
#                     roi = geometry.bounds
#                     gee_extent = ee.Geometry.Rectangle(roi)
#
#                     download_data = ee.ImageCollection(data).select(band).filterDate(start_date, end_date). \
#                         filterBounds(gee_extent).reduce(reducer).multiply(multiply_scale).toFloat()
#
#                     data_url = download_data.getDownloadURL({'name': data_name,
#                                                              'crs': 'EPSG:4269',  # NAD83
#                                                              'scale': 30,  # in meter. equal to ~0.02 deg
#                                                              'region': gee_extent,
#                                                              'format': 'GEO_TIFF'})
#                     key_word = data_name
#                     local_file_name = os.path.join(download_dir, f'{key_word}_{str(year)}_{str(month)}_{str(grid_sr)}.tif')
#                     print('Downloading', local_file_name, '.....')
#                     r = requests.get(data_url, allow_redirects=True)
#                     open(local_file_name, 'wb').write(r.content)
#
#                     # # This is a check block to see if downloaded datasets are OK
#                     # # sometimes a particular grid's data is corrupted but it's a random thing
#                     # # not sure why it happens. Re-downloading the same data might not have that error
#                     try:
#                         arr = read_raster_arr_object(local_file_name, get_file=False)
#
#                     except:
#                         print('Downloaded data corrupted. Re-downloading.....')
#                         print('Downloading', local_file_name, '.....')
#                         r = requests.get(data_url, allow_redirects=True)
#                         open(local_file_name, 'wb').write(r.content)
#
#                 # mosaic_name = f'{data_name}_{year}_{month}.tif'
#                 # mosaic_dir = os.path.join(download_dir, f'{merge_keyword}')
#                 # makedirs([mosaic_dir])
#                 # search_by = f'*{year}_{month}*.tif'
#                 # mosaic_rasters(download_dir, mosaic_dir, mosaic_name, ref_raster=refraster, search_by=search_by,
#                 #                nodata=no_data_value)
#                 # print(f'{data_name} monthly data downloaded and merged')
#
#             else:
#                 pass
#
#
# download_OpenET_data_monthly(data_name='OpenET_ensemble', download_dir='F:/WestUS_NetGW/scratch',
#                              year_list=[2016, 2017, 2018, 2019, 2020, 2021, 2022], month_range=(1, 12),
#                              merge_keyword=None,
#                              grid_shape='F:/WestUS_NetGW/scratch/openet_test.shp',
#                               refraster=WestUS_raster)