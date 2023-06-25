import os
import pickle
import numpy as np
import pandas as pd
from glob import glob
import dask.dataframe as ddf
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder

from Codes.utils.system_ops import makedirs, copy_file
from Codes.utils.raster_ops import read_raster_arr_object

no_data_value = -9999
WestUS_raster = '../../Data_main/Compiled_data/reference_rasters/Western_US_refraster_2km.tif'
model_res = 0.02000000000000000389  # in deg, 2 km


def reindex_df(df):
    """
    Reindex dataframe based on column names.

    :param df: Predictor dataframe.

    :return: Reindexed dataframe.
    """
    sorted_columns = sorted(df.columns)
    df = df.reindex(sorted_columns, axis=1)

    return df


def apply_OneHotEncoding(input_df):
    one_hot = OneHotEncoder()
    input_df_enc = one_hot.fit_transform(input_df)
    return input_df_enc


def create_dataframe_csv(input_data_dir, output_csv, search_by='*.tif', years=(2010, 2015),
                         drop_datasets=('MODIS_ET', 'MODIS_Terra_EVI', 'MODIS_Terra_NDVI'),
                         encode_cols=('USDA_cropland', 'USDA_developed'),
                         skip_dataframe_creation=False):
    """
    Create dataframe from predictor variables (raster).

    ** All variable names should start with capital letter. This is needed for keeping fips to the last during
       reindex().

    :param input_data_dir: Input rasters' directory.
    :param output_csv: Output csv file path.
    :param search_by: Input raster search criteria. Defaults to '*.tif'.
    :param years: Tuple/List of years to be included in predictor dataset. Default set to (2010, 2015). If only 1 year
                  use a list like [2015].
    :param drop_datasets: Tuple/List of variable names to drop while creating dataframe/csv.
    :param encode_cols: Tuple/List of variable names that need one hot encoding.
    :param skip_dataframe_creation: Set to True if want to skip processing.

    :return: A dataframe created where columns represent individual predictors.
    """
    makedirs([os.path.dirname(output_csv)])

    # Code block will be modified here for static predictors (we might/might not need timeframe at all)
    county_id_raster = '../Data_main/shapefiles/Western_US_ref_shapes/Western_US_countyID.tif'
    county_id_raster = copy_file(county_id_raster, input_data_dir, rename=None)

    if not skip_dataframe_creation:
        print('Creating Predictors csv...')
        predictors_list = glob(os.path.join(input_data_dir, search_by))
        predictors = []
        years_str = [str(y) for y in years]

        for prd in predictors_list:  # to add predictors of only selected years
            if any(yr in prd for yr in years_str):
                predictors.append(prd)

        predictors = sorted(predictors)
        predictor_dict = {}
        for year in years_str:
            for predictor in predictors:
                name = os.path.splitext(os.path.basename(predictor))[0]
                yr = name[name.rfind('_') + 1:]
                if yr == year:
                    variable_name = name[:name.rfind('_')]

                    if variable_name not in drop_datasets:  # predictors in drop_datasets will not be in predictor csv
                        raster_arr, file = read_raster_arr_object(predictor, get_file=True)
                        raster_arr = raster_arr.flatten()

                        if variable_name not in predictor_dict.keys():  # for first year
                            predictor_dict[variable_name] = raster_arr
                        else:  # for additional years
                            predictor_dict[variable_name] = np.append(predictor_dict[variable_name], raster_arr)
                    else:
                        pass

            if year == years_str[0]:  # for first year
                num_data_pixels_per_yr = len(predictor_dict[variable_name])
                predictor_dict['Year'] = [int(year) for i in range(num_data_pixels_per_yr)]

                # county ID
                name = os.path.splitext(os.path.basename(county_id_raster))[0]
                raster_arr = read_raster_arr_object(county_id_raster, get_file=False)
                raster_arr = raster_arr.flatten()
                predictor_dict[name] = raster_arr

            else:
                new_year = [int(year) for i in range(num_data_pixels_per_yr)]
                predictor_dict['Year'].extend(new_year)  # used extend. append adds the new years as a generator object

                # county ID
                name = os.path.splitext(os.path.basename(county_id_raster))[0]
                raster_arr = read_raster_arr_object(county_id_raster, get_file=False)
                raster_arr = raster_arr.flatten()
                predictor_dict[name] = np.append(predictor_dict[name], raster_arr)

        # # adding disaggregated (pynco) observed data
        gw_obsv_arr, _ = read_raster_arr_object('../Data_main/Compiled_data/pycno_gw.tif')
        predictor_dict['gw_observed'] = gw_obsv_arr.flatten()

        predictor_df = pd.DataFrame(predictor_dict)
        predictor_df = predictor_df.dropna(axis='index')

        # One-hot encoding
        if encode_cols is not None:
            for col in encode_cols:
                df_enc = pd.get_dummies(predictor_df[col], prefix=col)
                for i in df_enc.columns:    # adding encoded columns to the main dataframe
                    predictor_df[i] = df_enc[i]

        predictor_df = predictor_df.rename(columns={'Western_US_countyID': 'fips'})
        predictor_df = reindex_df(predictor_df)
        predictor_df = predictor_df.astype({'fips': 'int'})
        predictor_df = predictor_df.sort_values(by=['fips', 'Year'], ascending=True, axis=0)
        predictor_df.to_csv(output_csv, index=False)

        print('Predictors csv created\n')

    else:
        print('Loading Predictors csv...\n')

    return output_csv


def create_train_val_test_data(predictor_csv, observed_data_csv, data_fraction, train_fraction, val_fraction,
                               test_fraction, output_dir='../Data_main/Model_csv',
                               drop_columns=None, train_val_test_exists=False):
    """
    Create Train, Validation, and Test dataset (csv) from the predictor csv.

    :param predictor_csv: File path of predictor csv as input.
    :param observed_data_csv: File path of observed county-wise pumping data csv as input.
    :param data_fraction: Fraction of data to be used for model training, validating, and testing.
    :param train_fraction: Fraction of data (from use_data_fraction of data_csv) to be used for training.
    :param val_fraction: Fraction of data (from use_data_fraction of data_csv) to be used for validating.
    :param test_fraction: Fraction of data (from use_data_fraction of data_csv) to be used for testing.
    :param output_dir: File path of output directory to save/load train, validation, and test datasets.
    :param drop_columns: List of columns to drop from the predictor csv. Default set to None.
    :param train_val_test_exists: Set to True if want to skip train, validation, and test data creation again.
                                  Existing datasets will be loaded.

    :return: Train, Validation, and Test dataset variables and observed csv.
    """
    if not train_val_test_exists:
        print('Creating Train, Validation, Test dataset...')
        # # Processing observed dataset
        observed_df = pd.read_csv(observed_data_csv)
        observed_df = observed_df[(observed_df['total_gw_observed'] > 0) & (observed_df['total_gw_observed'] <250)]  # testing outlier removal by removing top and zero counties        print(len(observed_df))
        ###################
        # Keeping year 2010 as 2010. Replacing year 2015 as 3020, otherwise tensor faces trouble differentiating between
        # 2010 and 2015
        observed_df['Year'] = observed_df['Year'].apply(lambda x: 3020 if x == 2015 else 2010)
        ###################

        # Creating 'fips_years' attribute to use in train, val, test split
        observed_df = observed_df[['fips', 'Year', 'total_gw_observed']]
        observed_df['fips_years'] = observed_df['fips'].astype(str) + observed_df['Year'].astype(str)
        observed_df['fips_years'] = observed_df['fips_years'].astype(int)  # for sorting the data
        observed_df = observed_df.sort_values(by=['total_gw_observed'], ascending=False)

        # Creating fips_years_unique and selecting fips_years that will be splitted into train, validation, and
        # test data.
        fips_years_unique = observed_df['fips_years'].unique()
        fips_years_unique = list(np.random.permutation(fips_years_unique))

        train_length = round(len(fips_years_unique) * data_fraction * train_fraction)
        validation_length = round(len(fips_years_unique) * data_fraction * val_fraction)
        test_length = round(len(fips_years_unique) * data_fraction * test_fraction)

        # Selecting fips_years based on lengths
        train_fips = fips_years_unique[:train_length]
        validation_fips = fips_years_unique[train_length:(train_length + validation_length)]
        test_fips = fips_years_unique[
                    (train_length + validation_length):(train_length + validation_length + test_length)]

        # Top counties with high pumping has been added to the train, validation, and test fips. This is a very
        # important step for train dataset, to ensure that high pumping counties are included in model training
        top_pumping_counties = observed_df.iloc[0:50, :]['fips_years'].to_list()
        top_pumping_counties = list(np.random.permutation(top_pumping_counties))

        top_train_length = round(len(top_pumping_counties) * train_fraction)
        top_val_length = round(len(top_pumping_counties) * val_fraction)
        top_test_length = round(len(top_pumping_counties) * test_fraction)

        top_train_fips = top_pumping_counties[:top_train_length]
        top_val_fips = top_pumping_counties[top_train_length:(top_train_length + top_val_length)]
        top_test_fips = top_pumping_counties[(top_train_length + top_val_length):
                                             (top_train_length + top_val_length + top_test_length)]

        # Modified train, validation, and test fips_years with added fips_years with high pumping values
        train_fips = train_fips + [i for i in top_train_fips if i not in (train_fips + validation_fips + test_fips)]
        validation_fips = validation_fips + \
                          [i for i in top_val_fips if i not in (train_fips + validation_fips + test_fips)]
        test_fips = test_fips + [i for i in top_test_fips if i not in (train_fips + validation_fips + test_fips)]

        observed_df = observed_df.drop(columns=['fips', 'Year'])
        observed_df = observed_df.sort_values(by=['fips_years'], ascending=True, axis=0)

        train_observed = observed_df[observed_df['fips_years'].isin(train_fips)]
        validation_observed = observed_df[observed_df['fips_years'].isin(validation_fips)]
        test_observed = observed_df[observed_df['fips_years'].isin(test_fips)]

        train_observed = train_observed.sort_values(by=['fips_years'], ascending=True, axis=0)
        validation_observed = validation_observed.sort_values(by=['fips_years'], ascending=True, axis=0)
        test_observed = test_observed.sort_values(by=['fips_years'], ascending=True, axis=0)

        train_observed.to_csv(os.path.join(output_dir, 'train_obsv.csv'), index=False)
        validation_observed.to_csv(os.path.join(output_dir, 'validation_obsv.csv'), index=False)
        test_observed.to_csv(os.path.join(output_dir, 'test_obsv.csv'), index=False)

        print('Train, Validation, Test observed data created')

        # # Processing predictor dataset
        predictor_df = ddf.read_csv(predictor_csv)
        if drop_columns is not None:
                try:  # added try-except because dask uses 'labels' while pandas uses 'columns' keyword.
                    predictor_df = predictor_df.drop(columns=drop_columns, axis=1)
                except:
                    predictor_df = predictor_df.drop(labels=drop_columns, axis=1)

        ###################
        # Keeping year 2010 as 2010. Replacing year 2015 as 3020, otherwise tensor faces trouble differentiating between
        # 2010 and 2015
        predictor_df['Year'] = predictor_df['Year'].apply(lambda x: 3020 if x == 2015 else 2010, meta=('Year', 'int64'))
        ###################

        # Creating 'fips_years' attribute to use in train, val, test split
        predictor_df['fips_years'] = predictor_df['fips'].astype(str) + predictor_df['Year'].astype(str)
        predictor_df['fips_years'] = predictor_df['fips_years'].astype(int)  # for sorting the data

        # Selecting part of original csv based on randomly selected fip_years to create train, test, and validation set
        train_set = predictor_df[predictor_df['fips_years'].isin(train_fips)]
        validation_set = predictor_df[predictor_df['fips_years'].isin(validation_fips)]
        test_set = predictor_df[predictor_df['fips_years'].isin(test_fips)]

        # Merging predictor df with observed df to add county-level GW observed data in the predictor dataframe
        train_set = train_set.merge(train_observed, on='fips_years', how='left').reset_index()
        test_set = test_set.merge(test_observed, on='fips_years', how='left').reset_index()
        validation_set = validation_set.merge(validation_observed, on='fips_years', how='left').reset_index()

        # Dropping index column (that generated from merging)
        train_set = train_set.drop(columns=['index'])
        test_set = test_set.drop(columns=['index'])
        validation_set = validation_set.drop(columns=['index'])

        # Moving fips-years column to the end of df
        train_set['fips_years'] = train_set.pop('fips_years')
        test_set['fips_years'] = test_set.pop('fips_years')
        validation_set['fips_years'] = validation_set.pop('fips_years')

        # sorting based on fips_years to keep consistency (might not be required)
        train_set = train_set.sort_values(by=['fips_years'], ascending=True, axis=0)
        validation_set = validation_set.sort_values(by=['fips_years'], ascending=True, axis=0)
        test_set = test_set.sort_values(by=['fips_years'], ascending=True, axis=0)

        train_set.compute().to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)
        validation_set.compute().to_csv(os.path.join(output_dir, 'validation_data.csv'), index=False)
        test_set.compute().to_csv(os.path.join(output_dir, 'test_data.csv'), index=False)

        train_csv = os.path.join(output_dir, 'train_data.csv')
        val_csv = os.path.join(output_dir, 'validation_data.csv')
        test_csv = os.path.join(output_dir, 'test_data.csv')

        train_obsv = os.path.join(output_dir, 'train_obsv.csv')
        val_obsv = os.path.join(output_dir, 'validation_obsv.csv')
        test_obsv = os.path.join(output_dir, 'test_obsv.csv')

        print('Train, Validation, Test predictor dataset created \n')

    else:
        print('Loading Train/val/test data... \n')
        train_csv = os.path.join(output_dir, 'train_data.csv')
        val_csv = os.path.join(output_dir, 'validation_data.csv')
        test_csv = os.path.join(output_dir, 'test_data.csv')

        train_obsv = os.path.join(output_dir, 'train_obsv.csv')
        val_obsv = os.path.join(output_dir, 'validation_obsv.csv')
        test_obsv = os.path.join(output_dir, 'test_obsv.csv')

    return train_csv, val_csv, test_csv, train_obsv, val_obsv, test_obsv


def create_model_input(predictor_csv, observed_csv, drop_columns=('fips', 'Year')):
    """
    Generates numpy array of predictors and observed data, n_inputs number, and fips_years_array.
    Array of observed data and fip_years are only generated if use_for_testing is True.
    ** set use_for_testing = True for testing/predicting on new data.

    :param predictor_csv: File path of predictor csv.
    :param observed_csv: Default set to None. File path of observed data csv used for model testing.
                         Set a file path and use_for_testing=True for enabling this option.
    :param drop_columns: Tuple/List of data/columns to drop. Default set to ('fips', 'Year'). To drop
                                   additional data/columns add them with ('fips', 'Year').
    :param use_for_testing: Set to True if data will be used for model testing/predicting on new data.

    :return: if use_for_testing=False, returns -- predictor_arr, n_inputs
             if use_for_testing=True, returns -- predictor_arr, n_inputs, observed_arr, fips_years_arr
    """
    predictor_df = pd.read_csv(predictor_csv)
    fips_years_arr = predictor_df[['fips_years']].to_numpy()

    predictor_df = predictor_df.drop(columns=list(drop_columns))
    columns = list(predictor_df.columns)
    print(f'Using predictors: {columns[:-1]}')
    predictor_arr = predictor_df[columns].to_numpy()
    n_inputs = len(columns) - 3  # without disaggregated observed, total observed and fips_years columns

    observed_df = pd.read_csv(observed_csv)
    observed_arr = observed_df[['total_gw_observed']].to_numpy()

    return predictor_arr, n_inputs, observed_arr, fips_years_arr


def calculate_rmse(Y_pred, Y_obsv):
    """
    Calculates RMSE value of model prediction vs observed data.

    :param Y_pred: prediction array or panda series object.
    :param Y_obsv: observed array or panda series object.

    :return: RMSE value.
    """
    if isinstance(Y_pred, np.ndarray):
        Y_pred = Y_pred.reshape(-1, 1)
        Y_obsv = Y_obsv.reshape(-1, 1)
        rmse_val = np.sqrt(np.mean((Y_obsv - Y_pred) ** 2))
    else:  # in case of pandas series
        rmse_val = np.sqrt(np.mean((Y_obsv - Y_pred) ** 2))
    return rmse_val


def calculate_r2(Y_pred, Y_obsv):
    """
    Calculates R2 value of model prediction vs observed data.

    :param Y_pred: prediction array or panda series object.
    :param Y_obsv: observed array or panda series object.

    :return: R2 value.
    """
    if isinstance(Y_pred, np.ndarray):
        Y_pred = Y_pred.reshape(-1, 1)
        Y_obsv = Y_obsv.reshape(-1, 1)
        r2_val = r2_score(Y_obsv, Y_pred)
    else:  # in case of pandas series
        r2_val = r2_score(Y_obsv, Y_pred)
    return r2_val


def scatter_plot(Y_pred, Y_obsv, plot_name, savedir='../Model_Run/Plots'):
    """
    Makes scatter plot of model prediction vs observed data.

    :param plot_name:
    :param Y_pred: flattened prediction array.
    :param Y_obsv: flattened observed array.
    :param savedir: filepath to save the plot.

    :return: A scatter plot of model prediction vs observed data.
    """
    fig, ax = plt.subplots()
    ax.plot(Y_obsv, Y_pred, 'o')
    ax.plot([0, 1], [0, 1], '-r', transform=ax.transAxes)
    ax.set_xlabel('GW Observed (mm)')
    ax.set_ylabel('GW Predicted (mm)')

    r2_val = round(calculate_r2(Y_pred, Y_obsv), 3)
    ax.text(0.1, 0.9, s=f'R2={r2_val}', transform=ax.transAxes)

    makedirs([savedir])
    fig_loc = os.path.join(savedir, plot_name)
    fig.savefig(fig_loc, dpi=300)


def plot_rmse_trace(rmse_torch, savedir='../Model_Run/Plots'):
    """
    Makes plot of standardized calculate_rmse for each epoch.

    :param rmse_torch: A list of torch tensor holding calculate_rmse values.
    :param savedir: filepath to save the plot.

    :return: A plot of standardized calculate_rmse for each epoch.
    """
    rmse_trace = [i.cpu().detach().numpy().item() for i in rmse_torch]
    final_rmse = round(rmse_trace[-1], 4)

    # remove it later
    rmse_trace = [i for i in rmse_trace if i < 2]


    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(rmse_trace, '--o', color='lightblue')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Standardized RMSE')
    ax.text(0.7, 0.9, s=f'calculate_rmse={final_rmse}', transform=ax.transAxes)

    makedirs([savedir])
    fig_loc = savedir + '/rmse_trace.jpeg'
    fig.savefig(fig_loc, dpi=300)


# def model_performance(trained_model, predictor_csv, observed_csv, train_means, train_stds, plot_name,
#                       drop_columns=('fips', 'Year')):
#
#     predictor_arr, n_inputs, obsv_arr, fips_years_arr = \
#         create_model_input(predictor_csv=predictor_csv, observed_csv=observed_csv,
#                            drop_columns=drop_columns)
#
#     prediction_df = Neural_Net.predict(X=predictor_arr, fips_years_arr=fips_years_arr, trained_model=trained_model,
#                                        train_means=train_means, train_stds=train_stds)
#     prediction_df = prediction_df[['fips_years', 'gw prediction (mm)']]
#
#     obsv_df = pd.read_csv(observed_csv)
#     prediction_df = prediction_df.merge(obsv_df, on=['fips_years'], how='left').reset_index()
#
#     prediction_dir = '../Model_Run/Prediction'
#     makedirs([prediction_dir])
#     prediction_df.to_csv(os.path.join(prediction_dir, 'prediction.csv'), index=False)
#
#     rmse_value = calculate_rmse(Y_pred=prediction_df['gw prediction (mm)'], Y_obsv=prediction_df['total_gw_observed'])
#     r2_value = calculate_r2(Y_pred=prediction_df['gw prediction (mm)'], Y_obsv=prediction_df['total_gw_observed'])
#
#     scatter_plot(Y_pred=prediction_df['gw prediction (mm)'], Y_obsv=prediction_df['total_gw_observed'],
#                  savedir='../Model_Run/Plots', plot_name=plot_name)
#
#     print(f'Unstandardized RMSE value= {rmse_value}')
#     print(f'R2 value= {r2_value}')


