import os
import re
import sys
import pickle
import numpy as np
import pandas as pd
from glob import glob
from osgeo import gdal
import rasterio as rio
import geopandas as gpd

from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from Codes.utils.system_ops import makedirs
from Codes.utils.ml_ops import reindex_df
from Codes.utils.raster_ops import read_raster_arr_object, write_array_to_raster, create_multiband_raster

no_data_value = -9999
model_res = 0.01976293625031605786  # in deg, ~2 km
WestUS_shape = '../../Data_main/shapefiles/Western_US_ref_shapes/WestUS_states.shp'
WestUS_raster = '../../Data_main/reference_rasters/Western_US_refraster_2km.tif'
GEE_merging_refraster_large_grids = '../../Data_main/reference_rasters/GEE_merging_refraster_larger_grids.tif'


def filter_effective_precip_training_data(training_zone_shp, general_output_dir, refraster=WestUS_raster,
                                          resolution=model_res, skip_processing=False):
    """
    Filter the rainfed cropET training data by defined filters and bounding boxes.

    :param training_zone_shp: Input shapefile path of the training bounding box zones.
    :param general_output_dir: Filepath of general output directory. The final output dir will be configured inside
                                the function.
    :param refraster: Default set to Western US reference raster.
    :param resolution: Default set to model resolution.
    :param skip_processing: Set to True if want to skip processing.

    :return: None.
    """
    if not skip_processing:
        # year_list and months to process over
        # starting from 2008 as rainfed cropET dataset starts from 2008
        years = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020]
        months = list(range(4, 11))

        # training zone bounding box gdf
        training_zone_gdf = gpd.read_file(training_zone_shp)

        # reference raster
        ref_arr, ref_file = read_raster_arr_object(refraster)
        total_bounds = ref_file.bounds

        # primary and secondary output directory creation
        shape_temp_output_dir = os.path.join(general_output_dir, 'shapes')
        cropET_interim_shape_dir = os.path.join(general_output_dir, 'cropET_interim_for_shapes')
        final_filtered_cropET_dir = os.path.join(general_output_dir, 'final_filtered_cropET_for_training')

        makedirs([general_output_dir, shape_temp_output_dir, cropET_interim_shape_dir, final_filtered_cropET_dir])

        for id, geom in zip(training_zone_gdf['id'], training_zone_gdf['geometry']):
            # creating shapefile for each of the training boxes and rasterizing the shapefile such that values inside
            # the box/shapefile is 1 and outside the box/shapefile is no data for that respective box

            # creating polygon shapefile for each training zone
            zone_selected = training_zone_gdf[training_zone_gdf['id'] == id]

            zone_shapefile = os.path.join(shape_temp_output_dir, f'shape_{id}.shp')
            zone_selected.to_file(zone_shapefile)

            # rasterizing each training zone
            zone_raster = os.path.join(shape_temp_output_dir, f'zone_raster_{id}.tif')
            raster_options = gdal.RasterizeOptions(format='Gtiff', outputBounds=list(total_bounds),
                                                   outputType=gdal.GDT_Float32, xRes=resolution, yRes=resolution,
                                                   noData=no_data_value, burnValues=1, attribute=None, allTouched=True)
            gdal.Rasterize(destNameOrDestDS=zone_raster, srcDS=zone_shapefile, options=raster_options,
                           resolution=resolution)

        rainfed_cropET_dir = '../../Data_main/Raster_data/Rainfed_cropET/WestUS_monthly'
        rainfed_cropland_dir = '../../Data_main/Raster_data/Rainfed_cropland'
        irrigated_cropland_dir = '../../Data_main/Raster_data/Irrigated_cropland'
        usda_cdl_dir = '../../Data_main/Raster_data/USDA_CDL/WestUS_yearly'
        excess_ET_filter_dir = '../../Data_main/Raster_data/Excess_ET_filter'

        # collecting all rainfed cropET dataset
        rainfed_cropET_datasets = glob(os.path.join(rainfed_cropET_dir, '*.tif'))

        # going over each rainfed cropET data and applying the rasterized zone raster on it
        # this will create individual raster for each cropET raster for each bounding box
        for cropET in rainfed_cropET_datasets:
            cropET_arr, file = read_raster_arr_object(cropET)

            year = os.path.basename(cropET).split('_')[2]
            month = os.path.basename(cropET).split('_')[3].split('.')[0]

            rainfed_cropland_data = glob(os.path.join(rainfed_cropland_dir, f'*{year}*.tif'))[0]
            irrigated_cropland_data = glob(os.path.join(irrigated_cropland_dir, f'*{year}*.tif'))[0]
            cdl_data = glob(os.path.join(usda_cdl_dir, f'*{year}*.tif'))[0]
            excess_et_filter_data = glob(os.path.join(excess_ET_filter_dir, f'*{year}*.tif'))[0]

            rainfed_cropland_arr = read_raster_arr_object(rainfed_cropland_data, get_file=False)
            irrigated_cropland_arr = read_raster_arr_object(irrigated_cropland_data, get_file=False)
            cdl_arr = read_raster_arr_object(cdl_data, get_file=False)
            excess_et_arr = read_raster_arr_object(excess_et_filter_data, get_file=False)

            # grass/pasture lands with rainfed croplands or any rainfed cropland with no overlapping with irrigated croplands,
            # excess_et_filter = 1 in both cases

            # collecting the train zone rasters and looping over them to apply each train zone raster on a cropET raster
            zone_rasters = glob(os.path.join(shape_temp_output_dir, '*.tif'))

            for zone_ras in zone_rasters:
                zone_id = os.path.basename(zone_ras).split('_')[2].split('.')[0]
                zone_arr = read_raster_arr_object(zone_ras, get_file=False)

                filtered_cropET_arr = np.where(((cdl_arr == 176) & (rainfed_cropland_arr == 1) & (excess_et_arr == 1)) |
                                               ((rainfed_cropland_arr == 1) & (irrigated_cropland_arr == -9999) &
                                                (excess_et_arr == 1)), cropET_arr, -9999)

                # applying the zone raster on the filtered cropET so that everything outside the box becomes nodata
                final_cropET_arr = filtered_cropET_arr * zone_arr

                # saving the cropET raster for each year-month for each train zone
                output_cropET = os.path.join(cropET_interim_shape_dir, f'cropET_zone_{zone_id}_{year}_{month}.tif')
                write_array_to_raster(raster_arr=final_cropET_arr, raster_file=file, transform=file.transform,
                                      output_path=output_cropET)

        # compiling final filtered training data
        for year in years:
            for month in months:
                print(f'compiling filtered effect. precip training data for year {year}, month {month}...')

                # collecting data for all boxes for a single year-month
                # the  dot (.) is important to collect data for the same months. Not providing 'tif' here as glob will find
                # the rasters without it and we can't use both 'dot' and 'tif' for our purpose at the same time.
                filtered_training_rasters = glob(os.path.join(cropET_interim_shape_dir, f'*{year}_{month}.*'))
                final_train_arr = None  # we will iterate and paste values of each training box here

                # compiling the individual zone cropET raster in a single raster for each year-month
                for each in filtered_training_rasters:
                    if each == filtered_training_rasters[0]:
                        arr = read_raster_arr_object(each, get_file=False)
                        final_train_arr = arr

                    else:
                        arr = read_raster_arr_object(each, get_file=False)
                        final_train_arr = np.where(~np.isnan(final_train_arr), final_train_arr, arr)

                output_raster_name = os.path.join(final_filtered_cropET_dir, f'CropET_for_training_{year}_{month}.tif')
                write_array_to_raster(raster_arr=final_train_arr, raster_file=ref_file, transform=ref_file.transform,
                                      output_path=output_raster_name)
    else:
        pass


def create_monthly_dataframes_for_eff_precip_prediction(years_list, month_range,
                                                        monthly_data_path_dict, yearly_data_path_dict,
                                                        static_data_path_dict, datasets_to_include, output_dir,
                                                        skip_processing=False):
    """
    Create monthly dataframes of predictors to generate monthly effective prediction.

    :param years_list: A list of year_list for which data to include in the dataframe.
    :param month_range: A tuple of start and end month for which data to filter. Set to None if there is no monthly dataset.
    :param monthly_data_path_dict: A dictionary with monthly variables' names as keys and their paths as values.
    :param yearly_data_path_dict: A dictionary with static variables' names as keys and their paths as values.
                                  Set to None if there is static dataset.
    :param static_data_path_dict: A dictionary with yearly variables' names as keys and their paths as values.
                                  Set to None if there is no yearly dataset.
    :param datasets_to_include: A list of datasets to include in the dataframe.
    :param output_dir: Filepath of output directory.
    :param skip_processing: Set to True to skip this dataframe creation process.

    :return: None
    """
    if not skip_processing:
        makedirs([output_dir])

        month_list = [m for m in range(month_range[0], month_range[1] + 1)]  # creating list of months

        for year in years_list:  # 1st loop controlling year_list
            for month in month_list:  # 2nd loop controlling months
                print(f'creating dataframe for prediction - year={year}, month={month}...')

                variable_dict = {}

                # reading monthly data and storing it in a dictionary
                for var in monthly_data_path_dict.keys():
                    if var in datasets_to_include:
                        monthly_data = glob(os.path.join(monthly_data_path_dict[var], f'*{year}_{month}.tif*'))[0]
                        data_arr = read_raster_arr_object(monthly_data, get_file=False).flatten()

                        data_arr[np.isnan(data_arr)] = 0  # setting nan-position values with 0
                        variable_dict[var] = list(data_arr)
                        variable_dict['month'] = [int(month)] * len(data_arr)

                # reading monthly data and storing it in a dictionary
                if yearly_data_path_dict is not None:
                    for var in yearly_data_path_dict.keys():
                        if var in datasets_to_include:
                            yearly_data = glob(os.path.join(yearly_data_path_dict[var], f'*{year}*.tif'))[0]
                            data_arr = read_raster_arr_object(yearly_data, get_file=False).flatten()

                            data_arr[np.isnan(data_arr)] = 0  # setting nan-position values with 0
                            variable_dict[var] = list(data_arr)

                # reading static data and storing it in a dictionary
                if static_data_path_dict is not None:
                    for var in static_data_path_dict.keys():
                        if var in datasets_to_include:
                            static_data = glob(os.path.join(static_data_path_dict[var], '*.tif'))[0]
                            data_arr = read_raster_arr_object(static_data, get_file=False).flatten()

                            data_arr[np.isnan(data_arr)] = 0  # setting nan-position values with 0
                            variable_dict[var] = list(data_arr)

                predictor_df = pd.DataFrame(variable_dict)
                predictor_df = predictor_df.dropna()

                if 'Rainfed_Frac' in predictor_df.columns:  # renaming 'Rainfed_Frac' as this attribute will not be
                    # available for Irrigated pixels
                    predictor_df = predictor_df.rename(columns={'Rainfed_Frac': 'Cropland_Frac'})

                # saving input predictor csv
                monthly_output_csv = os.path.join(output_dir, f'predictors_{year}_{month}.csv')
                predictor_df.to_csv(monthly_output_csv, index=False)

    else:
        pass


def create_nan_pos_dict_for_irrigated_cropET(irrigated_cropET_dir, output_dir, skip_processing=False):
    if not skip_processing:
        makedirs([output_dir])

        print('creating nan position dicts for irrigated cropET')

        irrigated_cropET_datasets = glob(os.path.join(irrigated_cropET_dir, '*.tif'))

        nan_pos_dict = {}  # an empty dict where nan positions will be saved as boolean data
        for data in irrigated_cropET_datasets:
            name = os.path.basename(data).split('.')[0]
            year = os.path.basename(data).split('_')[2]
            month = os.path.basename(data).split('_')[3].split('.')[0]
            arr = read_raster_arr_object(data, get_file=False).flatten()
            nan_pos_dict[name] = np.isnan(arr)

            nan_pos_dict_name = os.path.join(output_dir, f'nan_pos_{year}_{month}.pkl')
            pickle.dump(nan_pos_dict, open(nan_pos_dict_name, mode='wb+'))
    else:
        pass


def create_monthly_effective_precip_rasters(trained_model, input_csv_dir, exclude_columns,
                                            irrig_cropET_nan_pos_dir, ref_raster,
                                            prediction_name_keyword, output_dir, skip_processing=False):
    if not skip_processing:
        makedirs([output_dir])

        # ref raster shape
        ref_arr, ref_file = read_raster_arr_object(ref_raster)
        ref_shape = ref_arr.shape

        # creating prediction raster for each month
        input_csvs = glob(os.path.join(input_csv_dir, '*.csv'))

        for csv in input_csvs:
            year = os.path.basename(csv).split('_')[1]
            month = os.path.basename(csv).split('_')[2].split('.')[0]
            print(f'Generating {prediction_name_keyword} prediction raster for year {year}, month {month}...')

            # loading input variable dataframe and filtering out columns
            df = pd.read_csv(csv)
            df = df.drop(columns=exclude_columns)
            df = reindex_df(df)

            # generating prediction with trained model
            pred_arr = trained_model.predict(df)
            pred_arr = np.array(pred_arr)

            # replacing values with -9999 where irrigated cropET is nan
            irrig_cropET_nan = glob(os.path.join(irrig_cropET_nan_pos_dir, f'*{year}_{month}.pkl*'))[0]
            nan_pos_dict = pickle.load(open(irrig_cropET_nan, mode='rb'))

            nan_key = f'Irrigated_cropET_{year}_{month}'
            pred_arr[nan_pos_dict[nan_key]] = ref_file.nodata

            # reshaping the prediction raster for Western US and saving
            pred_arr = pred_arr.reshape(ref_shape)
            # print(prediction_arr.shape)
            output_prediction_raster = os.path.join(output_dir, f'{prediction_name_keyword}_{year}_{month}.tif')
            write_array_to_raster(raster_arr=pred_arr, raster_file=ref_file, transform=ref_file.transform,
                                  output_path=output_prediction_raster)

    else:
        pass


def sum_monthly_effective_precip_to_grow_season(years_list, monthly_effective_precip_dir,
                                                irrigated_cropET_dir,
                                                grow_season_effective_precip_output_dir,
                                                skip_processing=False):
    """
    Sum monthly effective precip predictions by the ML model to grow season effective precip.

    :param years_list: A list of year_list to process the data.
    :param monthly_effective_precip_dir: Directory path of model-generated effective precipitation rasters.
    :param irrigated_cropET_dir: Directory path of growing season irrigated cropET. Will be used to set 0 and no data
                                 values to effective precipitation dataset.
    :param grow_season_effective_precip_output_dir: Directory path of summed growing season effective
    precipitation rasters.
    :param skip_processing: Set to True to skip this process.

    :return: None.
    """
    if not skip_processing:
        makedirs([grow_season_effective_precip_output_dir])

        for year in years_list:
            print(f'Summing effective precipitation monthly raster for grow season {year}...')

            input_rasters = glob(os.path.join(monthly_effective_precip_dir, f'*{year}*.tif'))

            sum_arr, file = None, None
            for raster in input_rasters:
                if raster == input_rasters[0]:
                    arr, file = read_raster_arr_object(raster)
                    sum_arr = arr
                else:
                    arr = read_raster_arr_object(raster, get_file=False)
                    sum_arr = np.nansum(np.dstack((sum_arr, arr)), axis=2)

            # # setting values over non-irrigated croplands to 0
            irrig_cropET = glob(os.path.join(irrigated_cropET_dir, f'*{year}*.tif'))[0]
            irrig_cropET_arr = read_raster_arr_object(irrig_cropET, get_file=False)

            sum_arr = np.where(np.isnan(irrig_cropET_arr), -9999, sum_arr)

            summed_raster = os.path.join(grow_season_effective_precip_output_dir, f'effective_precip_{year}.tif')
            write_array_to_raster(raster_arr=sum_arr, raster_file=file, transform=file.transform,
                                  output_path=summed_raster)


def dynamic_sum_peff(year_list, growsing_season_dir, monthly_peff_dir, gs_peff_dir, skip_processing=False):
    """
    Dynamically (spati-temporally) sums effective precipitation monthly rasters for the growing seasons.

    :param year_list: List of years to process the data for.
    :param growsing_season_dir: Directory path for growing season datasets.
    :param monthly_peff_dir:  Directory path for monthly effective precipitation datasets.
    :param gs_peff_dir:  Directory path (output) for summed growing season effective precipitation datasets.
    :param skip_processing: Set to True if want to skip processing this step.
    :return:
    """
    if not skip_processing:
        makedirs([gs_peff_dir])

        # The regex r'_([0-9]{1,2})\.tif' extracts the month (1 or 2 digits; e.g., '_1.tif', '_12.tif')
        # from the filenames using the first group ([0-9]{1,2}).
        # The extracted month is then (inside the for loop in the sorting block) converted to an integer with int(group(1))
        # for proper sorting by month.
        month_pattern = re.compile(r'_([0-9]{1,2})\.tif')

        for year in year_list:
            # gathering and sorting the peff datasets by month (from 1 to 12)
            peff_datasets = glob(os.path.join(monthly_peff_dir, f'*{year}*.tif'))
            sorted_peff_datasets = sorted(peff_datasets, key=lambda x: int(
                month_pattern.search(x).group(1)))  # First capturing group (the month)

            # peff monthly array stacked in a single numpy array
            peff_arrs = np.stack([read_raster_arr_object(i, get_file=False) for i in sorted_peff_datasets], axis=0)

            # gathering, reading, and stacking growing season array
            gs_data = glob(os.path.join(growsing_season_dir, f'*{year}*.tif'))[0]
            start_gs_arr, ras_file = read_raster_arr_object(gs_data, band=1, get_file=True)
            end_gs_arr = read_raster_arr_object(gs_data, band=1, get_file=False)

            # We create a 1 pixel "kernel", representing months 1 to 12 (shape - 12, 1, 1).
            # Then it is broadcasted across the array and named as the kernel_mask.
            # The kernel_mask acts as a mask, and only sum peff values for months that are 'True'.
            kernel = np.arange(1, 13, 1).reshape(12, 1, 1)
            kernel_mask = (kernel >= start_gs_arr) & (kernel <= end_gs_arr)

            # sum peff arrays over the valid months using the kernel_mask
            summed_peff = np.sum(peff_arrs * kernel_mask, axis=0)

            # saving the summed peff array
            output_name = f'effective_precip_{year}.tif'
            output_path = os.path.join(gs_peff_dir, output_name)
            with rio.open(
                    output_path,
                    'w',
                    driver='GTiff',
                    height=summed_peff.shape[0],
                    width=summed_peff.shape[1],
                    dtype=np.float32,
                    count=1,
                    crs=ras_file.crs,
                    transform=ras_file.transform,
                    nodata=-9999
            ) as dst:
                dst.write(summed_peff, 1)
            break


def process_monthly_peff_rasters_to_multiband_forGEE(years, peff_monthly_dir, output_dir, nodata=no_data_value):
    """
    Compiles monthly effective precipitation estimates into multi-band rasters for each year.

    :param years: List of years.
    :param peff_monthly_dir: Filepath of monthly effective precipitation rasters.
    :param output_dir: Filepath of output directory to save the multi-band raster.
    :param nodata: Default set to -9999.

    :return: None,
    """
    makedirs([output_dir])
    for year in years:
        peff_data_list = []

        # collecting the monthly Peff estimates serially for a year
        for month in list(range(4, 11)):  # list of months from 4-10
            monthly_peff = glob(os.path.join(peff_monthly_dir, f'*{year}_{month}*.tif'))
            peff_data_list.append(monthly_peff[0])

        # creating the multi-band image for monthly datasets within an year
        output_raster = os.path.join(output_dir, f'effective_precip_{year}_monthly.tif')

        create_multiband_raster(input_files_list=peff_data_list,
                                output_file=output_raster, nodata=nodata)
        print(f'created effective precipitation monthly multi-band raster for {year}...')