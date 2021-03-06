# New York City Taxi Trip Duration
# Share code and data to improve ride time predictions
# Data set:
# https://www.kaggle.com/c/nyc-taxi-trip-duration/data
# https://www.kaggle.com/cabaki/knycmetars2016
# https://www.kaggle.com/oscarleo/new-york-city-taxi-with-osrm
# https://www.kaggle.com/mathijs/weather-data-in-new-york-city-2016 => Not use

# install: http://jupyter-contrib-nbextensions.readthedocs.io/en/latest/install.html
# install: https://github.com/Jupyter-contrib/jupyter_nbextensions_configurator
# install GDAL: https://sandbox.idre.ucla.edu/sandbox/tutorials/installing-gdal-for-windows
# install osmnx: http://geoffboeing.com/2014/09/using-geopandas-windows/
# Credit to:
# https://www.kaggle.com/ankasor/driving-distance-using-open-street-maps-data/notebook
# https://www.kaggle.com/gaborfodor/from-eda-to-the-top-lb-0-368/notebook
# https://www.kaggle.com/headsortails/nyc-taxi-eda-update-the-fast-the-classified/notebook
# https://www.kaggle.com/onlyshadow/a-practical-guide-to-ny-taxi-data-0-379/notebook
# https://www.kaggle.com/misfyre/stacking-model-378-lb-375-cv

# System
import datetime as dtime
from datetime import datetime
import sys
from inspect import getsourcefile
import os.path
import re
import time
from profilehooks import timecall
import csv
import subprocess
import os
import logging
import copy
import gc
import psutil
# from memory_profiler import profile
import glob
# data processing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from math import sqrt
from scipy import stats
from scipy.stats import norm
import math
from math import radians, cos, sin, asin, sqrt
from numpy import sort

# ML
# # Scikit-learn
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score, train_test_split, learning_curve, validation_curve, KFold
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.decomposition import PCA
# # XGB
from xgboost.sklearn import XGBRegressor
import xgboost as xgb
from xgboost import plot_importance
# LightGBM
from lightgbm import LGBMRegressor
import lightgbm as lgb
# Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import SGD, Adam
if os.name != 'nt':
    # CatBoost
    from catboost import Pool, CatBoostRegressor
    # Vowpal Wabbit
    from vowpalwabbit.sklearn_vw import VWRegressor
    from vowpalwabbit.sklearn_vw import tovw

pd.options.display.float_format = '{:,.4f}'.format
# Input data files are available in the DATA_DIR directory.
DATA_DIR = "data-temp"
LABEL = 'trip_duration'
N_CLUSTERS = 200  # Kmeans number of cluster
# Model choice
XGB = 1
VW = 2
CATBOOST = 3
LIGHTGBM = 4
ETREE = 5  # ExtraTreesRegressor
RTREE = 6  # RandomForestRegressor
DTREE = 7  # DecisionTreeRegressor
KERAS = 8  # Keras
# Learning param
# 'learning_rate': 0.1, 'min_child_weight': 1, 'max_depth': 10 => Best
# 'learning_rate': 0.1, 'min_child_weight': 5, 'max_depth': 10
# 'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 10
# 'max_depth': 5, 'learning_rate': 0.1, 'min_child_weight': 5
# {'max_depth': 10, 'colsample_bytree': 0.9, 'min_child_weight': 1}
LEARNING_RATE = 0.1
MIN_CHILD_WEIGHT = 1
MAX_DEPTH = 10
COLSAMPLE_BYTREE = 0.9

N_FOLDS = 5
# N_FOLDS = 2  # Use for testing
N_ROUNDS = 20000
# N_ROUNDS = 10  # Use for testing

KERAS_LEARNING_RATE = 0.1
KERAS_N_ROUNDS = 50
KERAS_BATCH_SIZE = 10
KERAS_NODES = 10
KERAS_LAYERS = 10
KERAS_DROPOUT_RATE = 0.2
LOG_LEVEL = logging.DEBUG
VERBOSE = True
SILENT = False


class TaxiTripDuration():
    def __init__(self, label, model_choice=XGB):
        self.label = label
        self.model_choice = model_choice

    @timecall
    def load_data(self):
        logger.info("Loading data ....")
        label = self.label
        # Load data. Download
        # from:https://www.kaggle.com/c/nyc-taxi-trip-duration/data
        train_data = pd.read_csv(DATA_DIR + "/train.csv")
        eval_data = pd.read_csv(DATA_DIR + "/test.csv")
        logger.debug("train size:" + str(train_data.shape) +
                     " test size:" + str(eval_data.shape))
        train_osm, eval_osm = self.load_traffic_orsm()
        logger.debug("Merging  2 data sets ...")
        col_use = ['id', 'total_distance', 'total_travel_time',
                   'number_of_steps',
                   'starting_street', 'end_street', 'step_maneuvers', 'step_direction']
        train_osm_data = train_osm[col_use]
        eval_osm_data = eval_osm[col_use]
        train_data = train_osm_data.join(train_data.set_index('id'), on='id')
        # Cleanup data => temporarily removed
        eval_data = eval_osm_data.join(eval_data.set_index('id'), on='id')
        logger.debug("Atter merging: train size:" + str(train_data.shape) +
                     " test size:" + str(eval_data.shape))
        train_data = self.cleanup_data(train_data)
        features = eval_data.columns.values
        self.target = train_data[label]
        self.combine_data = pd.concat(
            [train_data[features], eval_data], keys=['train', 'eval'])
        self.convert_datetime()
        # Load weather data from:
        # https://www.kaggle.com/mathijs/weather-data-in-new-york-city-2016 => Not use
        # self.load_and_combine_weather_data()
        # Load weather data from:
        # https://www.kaggle.com/cabaki/knycmetars2016
        self.load_and_combine_weather_data_metar()

        logger.debug("combine data:" + str(len(self.combine_data)))
        features = self.combine_data.columns.values
        logger.debug("Original features:" + str(len(features)))
        logger.debug(features)
        logger.info("Data loaded")

    @timecall
    def load_traffic_orsm_old(self):
        # Download from:
        # https://www.kaggle.com/oscarleo/new-york-city-taxi-with-osrm
        logger.info("Loading OSRM old data ...")
        train_osm = pd.read_csv(DATA_DIR + "/fastest_route_train.csv")
        eval_osm = pd.read_csv(DATA_DIR + "/fastest_route_test.csv")
        logger.debug("train_osm size:" + str(train_osm.shape) +
                     " test osm size:" + str(eval_osm.shape))
        return train_osm, eval_osm

    @timecall
    def load_traffic_orsm(self):
        # Download from:
        # https://www.kaggle.com/oscarleo/new-york-city-taxi-with-osrm
        logger.info("Loading OSRM data ...")
        fr1 = pd.read_csv(DATA_DIR + '/fastest_routes_train_part_1.csv')
        fr2 = pd.read_csv(DATA_DIR + '/fastest_routes_train_part_2.csv')
        train_osm = pd.concat((fr1, fr2))
        eval_osm = pd.read_csv(DATA_DIR + "/fastest_routes_test.csv")

        logger.debug("train_osm size:" + str(train_osm.shape) +
                     " test osm size:" + str(eval_osm.shape))
        return train_osm, eval_osm

    @timecall
    def load_and_combine_weather_data_old(self):
        logger.debug("Loading weather data ..")
        weather_data = pd.read_csv(
            DATA_DIR + "/weather_data_nyc_centralpark_2016.csv")
        logger.debug("Weather data len:" + str(len(weather_data)))
        # Convert date string to date_obj
        weather_data.loc[:, 'date_obj'] = pd.to_datetime(
            weather_data['date'], dayfirst=True).dt.date
        # convert object columns
        # logger.debug("Convert object columns")
        # col = 'precipitation'
        # T_value = 0.01
        # weather_data.loc[weather_data[col] == 'T'] = T_value
        # weather_data.loc[:, col] = weather_data[col].astype(float)

        # col = 'snow fall'
        # weather_data.loc[weather_data[col] == 'T'] = T_value
        # weather_data.loc[:, col] = weather_data[col].astype(float)

        # col = 'snow depth'
        # weather_data.loc[weather_data[col] == 'T'] = T_value
        # weather_data.loc[:, col] = weather_data[col].astype(float)
        # weather_data.drop('date', axis=1, inplace=True)

        weather_data = weather_data.set_index('date_obj')
        cols = ['maximum temerature',
                'minimum temperature', 'average temperature']
        weather_data = weather_data[cols]
        # Convert combine_data datetime string to date_obj => join with weather
        logger.debug("Merging weather data with train set ... ")
        self.combine_data.loc[:, 'date_obj'] = pd.to_datetime(
            self.combine_data['pickup_datetime'], format="%Y-%m-%d").dt.date
        # Join combine_data and weather_data
        self.combine_data = self.combine_data.join(
            weather_data, on='date_obj')

        # Drop un-used cols
        self.combine_data.drop('date_obj', axis=1, inplace=True)
        logger.debug("Done merging...")

    @timecall
    def load_and_combine_weather_data_metar(self):
        logger.debug("Loading weather data from KNYC Metar 2016..")
        weather = pd.read_csv(
            DATA_DIR + "/KNYC_Metars.csv")
        logger.debug("Weather data len:" + str(len(weather)))
        weather.loc[:, 'snow'] = 1 * (weather.Events.str.contains('Snow'))
        weather.loc[:, 'heavy_snow'] = 1 * (weather.Conditions == 'Heavy Snow')
        weather.loc[:, 'rain'] = 1 * (weather.Events.str.contains('Rain'))
        weather.loc[:, 'heavy_rain'] = 1 * (weather.Conditions == 'Heavy Rain')
        weather.loc[:, 'fog'] = 1 * (weather.Events.str.contains('Fog'))
        weather.loc[:, 'datetime_obj'] = pd.to_datetime(weather['Time'])
        weather.loc[:, 'pickup_year'] = weather['datetime_obj'].dt.year
        weather.loc[:, 'pickup_month'] = weather['datetime_obj'].dt.month
        weather.loc[:, 'pickup_day'] = weather['datetime_obj'].dt.day
        weather.loc[:, 'pickup_hour'] = weather['datetime_obj'].dt.hour
        weather = weather[['pickup_year', 'pickup_month', 'pickup_day',
                           'pickup_hour',
                           'Temp.', 'Precip', 'Wind Speed',
                           'Dew Point', 'Visibility',
                           'snow', 'rain', 'fog', 'heavy_snow', 'heavy_rain']]
        # weather = weather[['pickup_year', 'pickup_month', 'pickup_day',
        #                    'pickup_hour', 'Temp.', 'snow']]

        # split train_set and eval_set
        train_set = self.combine_data.loc['train']
        eval_set = self.combine_data.loc['eval']
        # Join combine_data and weather_data
        logger.debug("Merging weather data with train set ... " +
                     str(len(train_set)))
        train_set = pd.merge(train_set, weather, on=[
            'pickup_year', 'pickup_month', 'pickup_day', 'pickup_hour'], how='left')
        logger.debug("train set merged " + str(len(train_set)) +
                     " " + str(len(self.target)))
        logger.debug("Merging weather data with eval set ... " +
                     str(len(eval_set)))
        eval_set = pd.merge(eval_set, weather, on=[
            'pickup_year', 'pickup_month', 'pickup_day', 'pickup_hour'], how='left')
        logger.debug("eval set merged " + str(len(eval_set)))
        logger.debug(" combine train_set and eval_set ... ")
        self.combine_data = pd.concat(
            [train_set, eval_set], keys=['train', 'eval'])

        # FillNAN
        data = self.combine_data
        data.loc[:, 'Temp.'].fillna(data['Temp.'].mean(), inplace=True)
        data.loc[:, 'Precip'].fillna(data['Precip'].mean(), inplace=True)
        data.loc[:, 'Wind Speed'].fillna(
            data['Wind Speed'].mean(), inplace=True)
        data.loc[:, 'Dew Point'].fillna(data['Dew Point'].mean(), inplace=True)
        data.loc[:, 'Visibility'].fillna(
            data['Visibility'].mean(), inplace=True)
        data.loc[:, 'snow'].fillna(0, inplace=True)
        data.loc[:, 'rain'].fillna(0, inplace=True)
        data.loc[:, 'fog'].fillna(0, inplace=True)
        data.loc[:, 'heavy_snow'].fillna(0, inplace=True)
        data.loc[:, 'heavy_rain'].fillna(0, inplace=True)

        logger.debug("Done merging...")
        # test
        # print(self.target.isnull().sum())
        # train_set.loc[:, self.label] = self.target.values
        # print(train_set[self.label].isnull().sum())
        # quit()

    @timecall
    # @profile
    def load_preprocessed_data(self):
        logger.info("Loading preprocessed data ....")
        label = self.label
        # Load data. Download
        # from:https://www.kaggle.com/c/nyc-taxi-trip-duration/data
        self.train_data = pd.read_csv(DATA_DIR + "/train_pre.csv")
        self.eval_data = pd.read_csv(DATA_DIR + "/test_pre.csv")
        logger.debug("train size:" + str(self.train_data.shape) +
                     " test size:" + str(self.eval_data.shape))
        # logger.debug(self.train_data.dtypes)
        # features = eval_data.columns.values
        # self.target = train_data[label]
        logger.info("Data loaded")

    def check_null_data(self, data=None):
        logger.info("Check for null data")
        if data is None:
            data = self.combine_data
        # Get high percent of NaN data
        null_data = data.isnull()
        total = null_data.sum().sort_values(ascending=False)
        percent = (null_data.sum() / null_data.count()
                   ).sort_values(ascending=False)
        missing_data = pd.concat(
            [total, percent], axis=1, keys=['Total', 'Percent'])
        high_percent_miss_data = missing_data[missing_data['Percent'] > 0]
        # logger.debug(missing_data)
        logger.debug(high_percent_miss_data)
        miss_data_cols = high_percent_miss_data.index.values
        return miss_data_cols

    def fillnan(self):
        logger.info("FillNaN ...")
        data = self.combine_data
        data.loc[:, 'total_distance'].fillna(1, inplace=True)
        data.loc[:, 'number_of_steps'].fillna(1, inplace=True)

    def cleanup_data(self, data):
        # trip_duration < 22*3600,
        # dist > 0 | (near(dist, 0) & trip_duration < 60),
        # jfk_dist_pick < 3e5 & jfk_dist_drop < 3e5,
        # trip_duration > 10,
        # speed < 100)
        size1 = len(data)
        logger.info("Cleanup data. Size before:" + str(size1))
        label = self.label
        # data = data[(data[label] < 22 * 3600) & (data[label] > 10)] # => underfit
        # data = data[(data[label] < 1000000)] # => stack: 0.3687232896
        # data = data[(data[label] > 0)]
        # data = data[(data[label] <= 86000)]
        # data = data[(data[label] < 24 * 3600)]
        data = data[(data[label] < 22 * 3600)]  # => Best
        size2 = len(data)
        logger.info("Finish cleanup. Size after:" + str(size2) +
                    " .Total removed:" + str(size1 - size2))
        return data

    def convert_datetime(self):
        logger.info("Convert datetime ...")
        data = self.combine_data
        data.loc[:, 'datetime_obj'] = pd.to_datetime(data['pickup_datetime'])
        data.loc[:, 'pickup_year'] = data['datetime_obj'].dt.year
        data.loc[:, 'pickup_month'] = data['datetime_obj'].dt.month
        data.loc[:, 'pickup_weekday'] = data['datetime_obj'].dt.weekday
        data.loc[:, 'pickup_day'] = data['datetime_obj'].dt.day
        data.loc[:, 'pickup_dayofyear'] = data['datetime_obj'].dt.dayofyear
        data.loc[:, 'pickup_hour'] = data['datetime_obj'].dt.hour
        data.loc[:, 'pickup_whour'] = data['pickup_weekday'] * \
            24 + data['pickup_hour']
        data.loc[:, 'pickup_minute'] = data['datetime_obj'].dt.minute

    def convert_store_and_fwd_flag(self):
        logger.info("Convert store_and_fwd_flag ...")
        data = self.combine_data
        col = 'store_and_fwd_flag'
        data_dict = {'Y': 1, 'N': 0}
        data_tf = data[col].map(data_dict)
        data.loc[:, col].update(data_tf)

    def convert_starting_street(self):
        logger.info("Convert starting_street ...")
        data = self.combine_data
        col = 'starting_street'
        data_not_null = data[data[col].notnull()]
        le = LabelEncoder()
        data_tf = le.fit_transform(data_not_null[col])
        col_tf = col + '_tf'
        # data_not_null.loc[:, col_tf] = data_tf
        data.loc[data[col].notnull(), col_tf] = data_tf

    def convert_end_street(self):
        logger.info("Convert end_street ...")
        data = self.combine_data
        col = 'end_street'
        data_not_null = data[data[col].notnull()]
        le = LabelEncoder()
        data_tf = le.fit_transform(data_not_null[col])
        col_tf = col + '_tf'
        # data_not_null.loc[:, col_tf] = data_tf
        data.loc[data[col].notnull(), col_tf] = data_tf

    @timecall
    def feature_starting_street(self):
        data = self.combine_data
        col = 'starting_street_tf'
        logger.info("Feature enginering ", col)
        data_zero = data[data[col].isnull()]
        data_not_zero = data[data[col].notnull()]
        features = ['pickup_longitude', 'pickup_latitude']
        X = data_not_zero[features]
        Y = data_not_zero[col]
        model_tf = LinearRegression(n_jobs=-1)
        model_tf.fit(X, Y)
        X_eval = data_zero[features]
        Y_pred = model_tf.predict(X_eval)
        data.loc[data[col].isnull(), col] = Y_pred

    @timecall
    def feature_end_street(self):
        data = self.combine_data
        col = 'end_street_tf'
        logger.info("Feature enginering ", col)
        data_zero = data[data[col].isnull()]
        data_not_zero = data[data[col].notnull()]
        features = ['dropoff_longitude', 'dropoff_latitude']
        X = data_not_zero[features]
        Y = data_not_zero[col]
        model_tf = LinearRegression(n_jobs=-1)
        model_tf.fit(X, Y)
        X_eval = data_zero[features]
        Y_pred = model_tf.predict(X_eval)
        data.loc[data[col].isnull(), col] = Y_pred

    # credit to:
    # https://stackoverflow.com/questions/15736995/how-can-i-quickly-estimate-the-distance-between-two-latitude-longitude-points
    def haversine(self, lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * asin(sqrt(a))
        km = 6367 * c
        return km

    # credit: Numpy haversine calculation
    # https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas
    def haversine_np(self, lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        All args must be of equal length.
        """
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * \
            np.cos(lat2) * np.sin(dlon / 2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        km = 6367 * c
        return km

    # Credit: Calculate manhattan distance based on haversine
    # https://www.kaggle.com/gaborfodor/from-eda-to-the-top-lb-0-368/notebook
    def manhattan_np(self, lon1, lat1, lon2, lat2):
        a = self.haversine_np(lon1, lat1, lon2, lat1)
        b = self.haversine_np(lon1, lat1, lon1, lat2)
        return a + b

    # calculate direction
    # credit: https://www.kaggle.com/gaborfodor/from-eda-to-the-top-lb-0-367
    def bearing_array(self, lng1, lat1, lng2, lat2):
        AVG_EARTH_RADIUS = 6371  # in km
        lng_delta_rad = np.radians(lng2 - lng1)
        lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
        y = np.sin(lng_delta_rad) * np.cos(lat2)
        x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * \
            np.cos(lat2) * np.cos(lng_delta_rad)
        return np.degrees(np.arctan2(y, x))

    @timecall
    def feature_haversine(self):
        logger.info("Feature engineering: haversine_distance")
        data = self.combine_data
        data.loc[:, 'haversine_distance'] = self.haversine_np(
            data['pickup_longitude'], data['pickup_latitude'],
            data['dropoff_longitude'], data['dropoff_latitude'])

    @timecall
    def feature_manhattan(self):
        logger.info("Feature engineering: manhattan_distance")
        data = self.combine_data
        data.loc[:, 'manhattan_distance'] = self.manhattan_np(
            data['pickup_longitude'], data['pickup_latitude'],
            data['dropoff_longitude'], data['dropoff_latitude'])

    # make new feature base on direction
    @timecall
    def feature_direction(self):
        logger.info("Feature engineering: direction based on lat & lon")
        data = self.combine_data
        data.loc[:, 'direction'] = self.bearing_array(
            data['pickup_longitude'], data['pickup_latitude'],
            data['dropoff_longitude'], data['dropoff_latitude'])

    def estimate_total_distance(self):
        logger.info("Estimating total_distance ... ")
        data = self.combine_data
        col1 = 'haversine_distance'
        col = 'total_distance'
        data_zero = data[(data[col] == 0) & (data[col1] > 0)].copy()
        data_not_zero = data[(data[col] > 0.) & (data[col] > 0.)]
        features = ['pickup_longitude', 'pickup_latitude',
                    'dropoff_longitude', 'dropoff_latitude',
                    'haversine_distance']
        X = data_not_zero[features]
        Y = np.log(data_not_zero[col])
        model_tf = LinearRegression(n_jobs=-1)
        model_tf.fit(X, Y)
        X_eval = data_zero[features]
        total_distance_log = model_tf.predict(X_eval)
        total_distance = np.exp(total_distance_log)
        data_zero.loc[:, col] = total_distance
        # Correct for too big value
        normal_data = data_zero[col][data_zero[col] < 50000]
        data_zero.loc[data_zero[col] > 50000, col] = normal_data.mean()
        data.loc[(data[col] == 0.) & (data[col1] > 0.), col] = data_zero[col]

    @timecall
    def feature_total_distance(self):
        logger.info("Feature engineering: total_distance")
        data = self.combine_data
        col1 = 'haversine_distance'
        col = 'total_distance'
        logger.debug("check for haversine_distance=0")
        data.loc[data[col1] == 0, col] = 0
        data.loc[data[col1] == 0, 'number_of_streets'] = 1
        logger.debug("check for total_distance=0 and haversine_distance=0")
        train_set = data.loc['train'].copy()
        train_set.loc[:, self.label] = self.target.values
        train_set.loc[(train_set[col] == 0.) & (
            train_set[col1] == 0.), self.label] = 0
        self.target = train_set[self.label]
        self.estimate_total_distance()

    def cal_speed(self, distance_col):
        col = 'speed'
        data_speed = self.combine_data.loc['train'].copy()
        data_speed.loc[:, self.label] = self.target.values
        data_speed.loc[:, col] = data_speed[distance_col] / \
            data_speed[self.label]
        data_speed.loc[:, col].fillna(data_speed[col].mean(), inplace=True)
        return data_speed

    def speed_mean_by_col(self, col, suffix, data_speed):
        logger.debug("Speed mean by " + col + " " + suffix)
        data = self.combine_data
        group_col = 'speed'
        data_grp = data_speed[[col, group_col]]
        data_st = data_grp.groupby(
            col, as_index=False).mean().sort_values(col).reset_index()
        data_sr = pd.Series(data_st[group_col], index=data_st[col])
        data_dict = data_sr.to_dict()
        col_mean = col + suffix
        data.loc[:, col_mean] = data[col].map(data_dict)
        data.loc[:, col_mean].fillna(data[col_mean].mean(), inplace=True)

    @timecall
    def feature_speed_mean(self):
        logger.info(
            "Calculating speed_mean by total_distance for each feature")
        distance_col = 'total_distance'
        suffix = '_speed_mean'
        data_speed = self.cal_speed(distance_col)
        col = 'pickup_hour'
        self.speed_mean_by_col(col, suffix, data_speed)
        col = 'pickup_weekday'
        self.speed_mean_by_col(col, suffix, data_speed)
        col = 'pickup_day'
        self.speed_mean_by_col(col, suffix, data_speed)
        col = 'pickup_month'
        self.speed_mean_by_col(col, suffix, data_speed)
        col = 'starting_street_tf'
        self.speed_mean_by_col(col, suffix, data_speed)
        col = 'end_street_tf'
        self.speed_mean_by_col(col, suffix, data_speed)
        col = 'number_of_steps'
        self.speed_mean_by_col(col, suffix, data_speed)

    @timecall
    def feature_hv_speed_mean(self):
        logger.info("Calculating speed_mean by haversine for each feature")
        distance_col = 'haversine_distance'
        suffix = '_hv_speed_mean'
        data_speed = self.cal_speed(distance_col)
        col = 'pickup_hour'
        self.speed_mean_by_col(col, suffix, data_speed)
        col = 'pickup_weekday'
        self.speed_mean_by_col(col, suffix, data_speed)
        col = 'pickup_day'
        self.speed_mean_by_col(col, suffix, data_speed)
        col = 'pickup_month'
        self.speed_mean_by_col(col, suffix, data_speed)
        col = 'starting_street_tf'
        self.speed_mean_by_col(col, suffix, data_speed)
        col = 'end_street_tf'
        self.speed_mean_by_col(col, suffix, data_speed)
        col = 'number_of_steps'
        self.speed_mean_by_col(col, suffix, data_speed)

    @timecall
    def duration_mean_by_col(self, col):
        logger.debug("Duration mean by " + col)
        col_speed_mean = col + '_speed_mean'
        col_duration_mean = col + '_duration_mean'
        data = self.combine_data
        data.loc[:, col_duration_mean] = data['total_distance'] / \
            data[col_speed_mean]
        data.loc[:, col_duration_mean].fillna(
            data[col_duration_mean].mean(), inplace=True)

    @timecall
    def haversine_duration_mean_by_col(self, col):
        logger.debug("Duration mean by " + col)
        col_speed_mean = col + '_hv_speed_mean'
        col_duration_mean = col + '_hv_duration_mean'
        data = self.combine_data
        data.loc[:, col_duration_mean] = data['haversine_distance'] / \
            data[col_speed_mean]

    @timecall
    def feature_duration_mean(self):
        logger.info(
            "Calculating duration_mean by total_distance for each feature")
        self.duration_mean_by_col('pickup_hour')
        self.duration_mean_by_col('pickup_weekday')
        self.duration_mean_by_col('pickup_day')
        self.duration_mean_by_col('pickup_month')

        self.duration_mean_by_col('starting_street_tf')
        self.duration_mean_by_col('end_street_tf')

        # logger.info("Calculating duration_mean by haversine for each feature")
        # self.haversine_duration_mean_by_col('pickup_hour')
        # self.haversine_duration_mean_by_col('pickup_weekday')
        # self.haversine_duration_mean_by_col('pickup_day')
        # self.haversine_duration_mean_by_col('pickup_month')

    @timecall
    def feature_distance_by_step(self):
        logger.info("Calculating total_distance/number_of_steps ...")
        data = self.combine_data
        col = 'distance_per_step'
        data.loc[:, col] = data['total_distance'] / data['number_of_steps']
        data.loc[:, col].fillna(data[col].mean(), inplace=True)

    @timecall
    def feature_haversine_distance_by_step(self):
        logger.info("Calculating haversine_distance/number_of_steps ...")
        data = self.combine_data
        col = 'hv_distance_per_step'
        data.loc[:, col] = data['haversine_distance'] / \
            data['number_of_steps']
        data.loc[:, col].fillna(data[col].mean(), inplace=True)

    # calculate number of turns based on step_maneuvers
    @timecall
    def feature_total_turns(self):
        logger.info("Calculating turns based on step_maneuvers ")
        col = 'step_maneuvers'
        data = self.combine_data
        turns = data[col].apply(lambda x: x.count('turn'))
        data.loc[:, 'turns'] = turns
        data.loc[:, 'turns'].fillna(0, inplace=True)

    # calculate number of right turns based on step_direction
    @timecall
    def feature_left_turns(self):
        logger.info("Calculating left turns based on step_direction ")
        col = 'step_direction'
        col_cal = 'left_turns'
        data = self.combine_data
        turns = data[col].apply(lambda x: x.count('left'))
        data.loc[:, col_cal] = turns
        data.loc[:, col_cal].fillna(0, inplace=True)

    # calculate number of right turns based on step_direction
    @timecall
    def feature_right_turns(self):
        logger.info("Calculating right turns based on step_direction ")
        col = 'step_direction'
        col_cal = 'right_turns'
        data = self.combine_data
        turns = data[col].apply(lambda x: x.count('right'))
        data.loc[:, col_cal] = turns
        data.loc[:, col_cal].fillna(0, inplace=True)

    # calculate delay between trip_duration and total_travel_time
    def cal_trip_delay(self):
        col = 'trip_delay'
        data = self.combine_data.loc['train'].copy()
        data.loc[:, self.label] = self.target.values
        data.loc[:, col] = data[self.label] - data['total_travel_time']
        data.loc[:, col].fillna(0, inplace=True)
        return data

    # calculate mean of trip_delay by column
    @timecall
    def trip_delay_mean_by_col(self, col, suffix, data_temp):
        logger.debug("trip_delay_mean by " + col + " " + suffix)
        data = self.combine_data
        group_col = 'trip_delay'
        data_grp = data_temp[[col, group_col]]
        data_st = data_grp.groupby(
            col, as_index=False).mean().sort_values(col).reset_index()
        data_sr = pd.Series(data_st[group_col], index=data_st[col])
        data_dict = data_sr.to_dict()
        col_mean = col + suffix
        data.loc[:, col_mean] = data[col].map(data_dict)
        data.loc[:, col_mean].fillna(data[col_mean].mean(), inplace=True)

    @timecall
    def feature_trip_delay_mean(self):
        logger.info("Calculating trip_delay_meanfor each feature")
        suffix = '_tdm'
        data_temp = self.cal_trip_delay()
        col = 'pickup_hour'
        self.trip_delay_mean_by_col(col, suffix, data_temp)
        col = 'pickup_weekday'
        self.trip_delay_mean_by_col(col, suffix, data_temp)
        col = 'pickup_day'
        self.trip_delay_mean_by_col(col, suffix, data_temp)
        col = 'pickup_month'
        self.trip_delay_mean_by_col(col, suffix, data_temp)
        col = 'starting_street_tf'
        self.trip_delay_mean_by_col(col, suffix, data_temp)
        col = 'end_street_tf'
        self.trip_delay_mean_by_col(col, suffix, data_temp)
        # col = 'number_of_steps'
        # self.trip_delay_mean_by_col(col, suffix, data_temp)

    # calculate different between total_distance and haversine_distance
    def feature_hv_distance_diff(self):
        data = self.combine_data
        col = 'hv_distance_diff'
        data.loc[:, col] = data['total_distance'] - data['haversine_distance']
        data.loc[:, col].fillna(data['total_distance'], inplace=True)

    # calculate clusters based on Kmeans
    @timecall
    def cal_cluster(self, data, n_clusters):
        batch_size = 10000
        mbk = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=batch_size,
                              n_init=10, max_no_improvement=10, verbose=0)
        mbk.fit(data)
        mbk_means_labels = mbk.predict(data)
        return mbk_means_labels

    @timecall
    def feature_cluster(self):
        data = self.combine_data
        logger.info("Cluster for starting_street_tf, end_street_tf")
        col_use = ['starting_street_tf', 'end_street_tf']
        col_cluster = 'street_cluster'
        clusters = self.cal_cluster(data[col_use], N_CLUSTERS)
        data.loc[:, col_cluster] = clusters

        logger.info("Cluster for pickup")
        col_use = ['pickup_latitude', 'pickup_longitude']
        col_cluster = 'pickup_cluster'
        clusters = self.cal_cluster(data[col_use], N_CLUSTERS)
        data.loc[:, col_cluster] = clusters

        logger.info("Cluster for dropoff")
        col_use = ['dropoff_latitude', 'dropoff_longitude']
        col_cluster = 'dropoff_cluster'
        clusters = self.cal_cluster(data[col_use], N_CLUSTERS)
        data.loc[:, col_cluster] = clusters

        logger.info("Cluster for center coordinate")
        col_use = ['center_latitude', 'center_longitude']
        col_cluster = 'center_cluster'
        clusters = self.cal_cluster(data[col_use], N_CLUSTERS)
        data.loc[:, col_cluster] = clusters

        # 'pickup_latitude', 'pickup_longitude'
        logger.info("Cluster for pickup location and pickup_hour")
        col_use = ['pickup_hour', 'pickup_latitude', 'pickup_longitude']
        col_cluster = 'hour_pickup_location_cluster'
        clusters = self.cal_cluster(data[col_use], N_CLUSTERS)
        data.loc[:, col_cluster] = clusters

        logger.info("Cluster for pickup location and pickup_whour")
        col_use = ['pickup_whour', 'pickup_latitude', 'pickup_longitude']
        col_cluster = 'whour_pickup_location_cluster'
        clusters = self.cal_cluster(data[col_use], N_CLUSTERS)
        data.loc[:, col_cluster] = clusters

        logger.info("Cluster for pickup location and pickup_weekday")
        col_use = ['pickup_weekday', 'pickup_latitude', 'pickup_longitude']
        col_cluster = 'weekday_pickup_location_cluster'
        clusters = self.cal_cluster(data[col_use], N_CLUSTERS)
        data.loc[:, col_cluster] = clusters

        logger.info("Cluster for pickup location and pickup_day")
        col_use = ['pickup_day', 'pickup_latitude', 'pickup_longitude']
        col_cluster = 'day_pickup_location_cluster'
        clusters = self.cal_cluster(data[col_use], N_CLUSTERS)
        data.loc[:, col_cluster] = clusters

        logger.info("Cluster for pickup location and pickup_dayofyear")
        col_use = ['pickup_dayofyear', 'pickup_latitude', 'pickup_longitude']
        col_cluster = 'dayofyear_pickup_location_cluster'
        clusters = self.cal_cluster(data[col_use], N_CLUSTERS)
        data.loc[:, col_cluster] = clusters

        # 'dropoff_latitude', 'dropoff_longitude'
        logger.info("Cluster for drop-off location and pickup_hour")
        col_use = ['pickup_hour', 'dropoff_latitude', 'dropoff_longitude']
        col_cluster = 'hour_dropoff_location_cluster'
        clusters = self.cal_cluster(data[col_use], N_CLUSTERS)
        data.loc[:, col_cluster] = clusters

        logger.info("Cluster for drop-off location and pickup_whour")
        col_use = ['pickup_whour', 'dropoff_latitude', 'dropoff_longitude']
        col_cluster = 'whour_dropoff_location_cluster'
        clusters = self.cal_cluster(data[col_use], N_CLUSTERS)
        data.loc[:, col_cluster] = clusters

        logger.info("Cluster for drop-off location and pickup_weekday")
        col_use = ['pickup_weekday', 'dropoff_latitude', 'dropoff_longitude']
        col_cluster = 'weekday_dropoff_location_cluster'
        clusters = self.cal_cluster(data[col_use], N_CLUSTERS)
        data.loc[:, col_cluster] = clusters

        logger.info("Cluster for drop-off location and pickup_day")
        col_use = ['pickup_day', 'dropoff_latitude', 'dropoff_longitude']
        col_cluster = 'day_dropoff_location_cluster'
        clusters = self.cal_cluster(data[col_use], N_CLUSTERS)
        data.loc[:, col_cluster] = clusters

        logger.info("Cluster for drop-off location and pickup_dayofyear")
        col_use = ['pickup_dayofyear', 'dropoff_latitude', 'dropoff_longitude']
        col_cluster = 'dayofyear_dropoff_location_cluster'
        clusters = self.cal_cluster(data[col_use], N_CLUSTERS)
        data.loc[:, col_cluster] = clusters

        # pickup + drop-off
        logger.info("Cluster for pickup + drop-off location and pickup_hour")
        col_use = ['pickup_hour', 'pickup_latitude',
                   'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        col_cluster = 'hour_location_cluster'
        clusters = self.cal_cluster(data[col_use], N_CLUSTERS)
        data.loc[:, col_cluster] = clusters

        logger.info("Cluster for pickup + drop-off location and pickup_whour")
        col_use = ['pickup_whour', 'pickup_latitude',
                   'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        col_cluster = 'whour_location_cluster'
        clusters = self.cal_cluster(data[col_use], N_CLUSTERS)
        data.loc[:, col_cluster] = clusters

        logger.info("Cluster for pickup + drop-off location and pickup_weekday")
        col_use = ['pickup_weekday', 'pickup_latitude',
                   'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        col_cluster = 'weekday_location_cluster'
        clusters = self.cal_cluster(data[col_use], N_CLUSTERS)
        data.loc[:, col_cluster] = clusters

        logger.info("Cluster for pickup + drop-off location and pickup_day")
        col_use = ['pickup_day', 'pickup_latitude',
                   'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        col_cluster = 'day_location_cluster'
        clusters = self.cal_cluster(data[col_use], N_CLUSTERS)
        data.loc[:, col_cluster] = clusters

        logger.info(
            "Cluster for pickup + drop-off location and pickup_dayofyear")
        col_use = ['pickup_dayofyear', 'pickup_latitude',
                   'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude']
        col_cluster = 'dayofyear_location_cluster'
        clusters = self.cal_cluster(data[col_use], N_CLUSTERS)
        data.loc[:, col_cluster] = clusters

        # center coordinate
        logger.info("Cluster for center coordinate and pickup_hour")
        col_use = ['pickup_hour', 'center_latitude', 'center_longitude']
        col_cluster = 'hour_center_cluster'
        clusters = self.cal_cluster(data[col_use], N_CLUSTERS)
        data.loc[:, col_cluster] = clusters

        logger.info("Cluster for center coordinate and pickup_whour")
        col_use = ['pickup_whour', 'center_latitude', 'center_longitude']
        col_cluster = 'whour_center_cluster'
        clusters = self.cal_cluster(data[col_use], N_CLUSTERS)
        data.loc[:, col_cluster] = clusters

        logger.info("Cluster for center coordinate and pickup_weekday")
        col_use = ['pickup_weekday', 'center_latitude', 'center_longitude']
        col_cluster = 'weekday_center_cluster'
        clusters = self.cal_cluster(data[col_use], N_CLUSTERS)
        data.loc[:, col_cluster] = clusters

        logger.info("Cluster for center coordinate and pickup_day")
        col_use = ['pickup_day', 'center_latitude', 'center_longitude']
        col_cluster = 'day_center_cluster'
        clusters = self.cal_cluster(data[col_use], N_CLUSTERS)
        data.loc[:, col_cluster] = clusters

        logger.info("Cluster for center coordinate and pickup_dayofyear")
        col_use = ['pickup_dayofyear', 'center_latitude', 'center_longitude']
        col_cluster = 'dayofyear_center_cluster'
        clusters = self.cal_cluster(data[col_use], N_CLUSTERS)
        data.loc[:, col_cluster] = clusters

        # total_distance
        # logger.info("Cluster for total_distance and pickup_hour")
        # col_use = ['pickup_hour', 'total_distance']
        # col_cluster = 'hour_distance_cluster'
        # clusters = self.cal_cluster(data[col_use], N_CLUSTERS)
        # data.loc[:, col_cluster] = clusters

        # logger.info("Cluster for total_distance and pickup_whour")
        # col_use = ['pickup_whour', 'total_distance']
        # col_cluster = 'whour_distance_cluster'
        # clusters = self.cal_cluster(data[col_use], N_CLUSTERS)
        # data.loc[:, col_cluster] = clusters

        # logger.info("Cluster for total_distance and pickup_weekday")
        # col_use = ['pickup_weekday', 'total_distance']
        # col_cluster = 'weekday_distance_cluster'
        # clusters = self.cal_cluster(data[col_use], N_CLUSTERS)
        # data.loc[:, col_cluster] = clusters

        # logger.info("Cluster for total_distance and pickup_day")
        # col_use = ['pickup_day', 'total_distance']
        # col_cluster = 'day_distance_cluster'
        # clusters = self.cal_cluster(data[col_use], N_CLUSTERS)
        # data.loc[:, col_cluster] = clusters

        # logger.info("Cluster for total_distance and pickup_dayofyear")
        # col_use = ['pickup_dayofyear', 'total_distance']
        # col_cluster = 'dayofyear_distance_cluster'
        # clusters = self.cal_cluster(data[col_use], N_CLUSTERS)
        # data.loc[:, col_cluster] = clusters

    # Calculate count by cluster
    @timecall
    def cal_cluster_count(self, col):
        logger.debug("calcuate cluster count by " + col)
        data = self.combine_data
        group_col = 'id'
        data_grp = data[[col, group_col]]
        data_st = data_grp.groupby(
            col, as_index=False).count().sort_values(col).reset_index()
        data_sr = pd.Series(data_st[group_col], index=data_st[col])
        data_dict = data_sr.to_dict()
        col_count = col + '_count'
        data.loc[:, col_count] = data[col].map(data_dict)
        data.loc[:, col_count].fillna(data[col_count].mean(), inplace=True)

    @timecall
    def feature_cluster_count(self):
        col = 'pk_hour_dropoff_location_cluster'
        logger.info("Cluster count for " + col)
        self.cal_cluster_count(col)

        col = 'pk_whour_dropoff_location_cluster'
        logger.info("Cluster count for " + col)
        self.cal_cluster_count(col)

        col = 'pk_weekday_dropoff_location_cluster'
        logger.info("Cluster count for " + col)
        self.cal_cluster_count(col)

        col = 'pk_hour_pickup_location_cluster'
        logger.info("Cluster count for " + col)
        self.cal_cluster_count(col)

        col = 'pk_whour_pickup_location_cluster'
        logger.info("Cluster count for " + col)
        self.cal_cluster_count(col)

        col = 'pk_weekday_pickup_location_cluster'
        logger.info("Cluster count for " + col)
        self.cal_cluster_count(col)

        col = 'pk_hour_location_cluster'
        logger.info("Cluster count for " + col)
        self.cal_cluster_count(col)

        col = 'pk_whour_location_cluster'
        logger.info("Cluster count for " + col)
        self.cal_cluster_count(col)

        col = 'pk_weekday_location_cluster'
        logger.info("Cluster count for " + col)
        self.cal_cluster_count(col)

    # Calculate PCA for location
    @timecall
    def feature_pca(self):
        logger.info("Feature engineering: PCA for location")
        data = self.combine_data
        coords = np.vstack((data[['pickup_latitude', 'pickup_longitude']].values,
                            data[['dropoff_latitude', 'dropoff_longitude']].values
                            ))
        logger.debug("PCA fiting ...")
        pca = PCA().fit(coords)
        logger.debug("Tranform PCA features ...")
        data.loc[:, 'pickup_pca0'] = pca.transform(
            data[['pickup_latitude', 'pickup_longitude']])[:, 0]
        data.loc[:, 'pickup_pca1'] = pca.transform(
            data[['pickup_latitude', 'pickup_longitude']])[:, 1]
        data.loc[:, 'dropoff_pca0'] = pca.transform(
            data[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
        data.loc[:, 'dropoff_pca1'] = pca.transform(
            data[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
        data.loc[:, 'pca_manhattan'] = np.abs(
            data['dropoff_pca1'] - data['pickup_pca1']) + np.abs(data['dropoff_pca0'] - data['pickup_pca0'])

    # calculate new corrination based on pickup, dropoff location
    @timecall
    def feature_coordinate(self):
        logger.info("Feature engineering: new coordinate")
        data = self.combine_data
        data.loc[:, 'center_latitude'] = (
            data['pickup_latitude'].values + data['dropoff_latitude'].values) / 2
        data.loc[:, 'center_longitude'] = (
            data['pickup_longitude'].values + data['dropoff_longitude'].values) / 2

    def drop_unused_cols(self):
        data = self.combine_data
        data.drop(['pickup_datetime', 'datetime_obj', 'starting_street',
                   'end_street', 'step_maneuvers', 'step_direction', 'pickup_year'], axis=1, inplace=True)

    @timecall
    def preprocess_data(self):
        logger.info("Preproccesing data ...")
        self.fillnan()
        # self.convert_datetime() => moved to load_data
        self.convert_starting_street()
        self.convert_end_street()
        self.convert_store_and_fwd_flag()
        self.feature_haversine()
        self.feature_manhattan()
        self.feature_left_turns()
        self.feature_right_turns()
        self.feature_total_turns()
        self.feature_direction()
        self.feature_pca()
        self.feature_coordinate()
        self.feature_cluster()

        # Expriment
        # self.feature_cluster_count()
        # Expriment
        #
        # self.feature_speed_mean()
        # self.feature_hv_speed_mean()
        # self.feature_trip_delay_mean()
        # self.feature_hv_distance_diff() => No score improvement
        # self.feature_total_turns()
        # self.feature_duration_mean()
        # self.feature_distance_by_step()
        # self.feature_haversine_distance_by_step()
        # There is no NaN starting_street and end_street => no need to feature enginering
        # self.feature_starting_street()
        # self.feature_end_street()

        # Drop unsed columns
        self.drop_unused_cols()
        features = self.combine_data.columns.values
        logger.debug("Engineered features:" + str(len(features)))
        logger.debug(features)
        # logger.debug(self.combine_data.dtypes)

        # Save preprocess data
        train_set = self.combine_data.loc['train'].copy()
        train_set.loc[:, self.label] = self.target.values
        logger.debug("Check null of label:" +
                     str(train_set[self.label].isnull().sum()))
        eval_set = self.combine_data.loc['eval']
        logger.debug("Saving train set ...")
        train_set.to_csv(DATA_DIR + '/train_pre.csv', index=False)
        logger.debug("Saving eval set ...")
        eval_set.to_csv(DATA_DIR + '/test_pre.csv', index=False)
        # Save to class variable for later use
        self.train_data = train_set
        self.eval_data = eval_set

    def rmsle(self, y, y_pred, log=False):
        assert len(y) == len(y_pred)
        terms_to_sum = 0
        if log:
            terms_to_sum = [(math.log(math.fabs(y_pred[i]) + 1) -
                             math.log(y[i] + 1)) ** 2.0 for i, pred in enumerate(y_pred)]
        else:
            terms_to_sum = [(math.fabs(y_pred[i]) - y[i]) **
                            2.0 for i, pred in enumerate(y_pred)]
        return (sum(terms_to_sum) * (1.0 / len(y))) ** 0.5

    @timecall
    def search_best_model_params(self):
        logger.info("Prepare data for  model")
        data = self.train_data
        target = data[self.label]
        target_log = np.log(target)
        train_set = data.drop(
            ['id', self.label], axis=1).astype(float)
        X_train, X_test, Y_train, Y_test = train_test_split(
            train_set, target_log, train_size=0.85, random_state=1234)
        # 'learning_rate': [0.1, 0.3],
        param_grid = {
            'learning_rate': [0.1, 0.03],
            "max_depth": [5, 10, 20],
            'min_child_weight':  [1, 5, 20],
        }
        model = XGBRegressor(n_estimators=200, learning_rate=0.1, n_jobs=-1)
        logger.debug("Searching for best params")
        scorer = make_scorer(self.rmsle, greater_is_better=False)
        grid_search = GridSearchCV(
            model, param_grid, n_jobs=1, cv=5, verbose=3, scoring=scorer)
        grid_search.fit(X_test.values, Y_test.values)
        logger.debug("Best params:" + str(grid_search.best_params_))
        logger.debug("Best score:" + str(grid_search.best_score_))

    def feature_correlation(self):
        logger.info("Feature correlation ...")
        data = self.combine_data.loc['train'].copy()
        data.loc[:, self.label] = self.target.values
        correlation = data.corr()[self.label].sort_values()
        logger.debug(str(correlation))

    def xgb_model(self, learning_rate=LEARNING_RATE, random_state=1000):
        model = XGBRegressor(n_estimators=N_ROUNDS,
                             max_depth=10,
                             learning_rate=learning_rate,
                             min_child_weight=1,
                             #  gamma=0,
                             random_state=random_state,
                             n_jobs=-1,
                             silent=SILENT
                             )
        return model

    def lgbm_model(self, random_state=1024):
        model = LGBMRegressor(objective='regression_l2',
                              metric='l2_root',
                              n_estimators=N_ROUNDS,
                              #   n_estimators=10,
                              learning_rate=0.01,
                              num_leaves=1024,
                              seed=random_state,
                              nthread=-1, silent=False)
        return model

    # calculate index of features
    def convert_to_categrorical_features(self, data):
        logger.info("Convert category features")
        cat_features = ['vendor_id', 'store_and_fwd_flag', 'pickup_month',
                        'pickup_weekday', 'pickup_day', 'pickup_hour',
                        'pickup_whour', 'pickup_minute'
                        ]
        cols = data.columns.values
        sidx = np.argsort(cols)
        categorical_features_indices = sidx[np.searchsorted(
            cols, cat_features, sorter=sidx)]
        logger.debug("Categories features:" + cat_features)
        logger.debug("Categories feature index:" +
                     categorical_features_indices)
        # logger.debug("Change categories feature type to int")
        # for col in cat_features:
        #     data.loc[:, col] = data[col].astype(int)
        # logger.debug(data.dtypes)
        return categorical_features_indices

    def vowpalwabbit_model(self):
        model = VWRegressor(learning_rate=0.1, quiet=False, passes=100)
        return model

    def catboost_model(self, random_state=648):
        model = CatBoostRegressor(
            iterations=N_ROUNDS,  # => overfit if iteration > 20k
            # iterations=10,
            od_pval=None,
            od_type="Iter",
            od_wait=150,
            learning_rate=0.1,
            depth=MAX_DEPTH,
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=random_state,
            use_best_model=True, train_dir=DATA_DIR + "/node_modules",
            verbose=VERBOSE)
        return model

    def etree_model(self, random_state=911):
        model = ExtraTreesRegressor(
            n_estimators=100,  #
            # n_estimators=10,  #
            max_depth=50,  # RMSLE: 0.345729987977
            random_state=random_state,
            n_jobs=-1,
            verbose=VERBOSE)
        return model

    def rtree_model(self, random_state=1177):
        model = RandomForestRegressor(
            n_estimators=100,  #
            # n_estimators=10,  #
            max_depth=50,  # RMSLE: 0.345309800425
            random_state=random_state,
            n_jobs=-1,
            verbose=VERBOSE)
        return model

    def dtree_model(self, random_state=7734):  # Bad score
        model = DecisionTreeRegressor(
            # max_depth=50,  # RMSLE: 0.345309800425
            random_state=random_state,
        )
        return model

    def keras_model(self, random_state=9999):
        # load train_data
        data = self.train_data
        n_features = len(data.columns) - 2
        decay = KERAS_LEARNING_RATE / KERAS_N_ROUNDS
        logger.debug("n_features:" + str(n_features))

        # create model
        model = Sequential()
        nodes = n_features // 2
        model.add(Dense(n_features, input_dim=n_features,
                        kernel_initializer='normal', activation='relu'))
        for i in range(KERAS_LAYERS):
            model.add(Dense(nodes,
                            kernel_initializer='normal', activation='relu'))
            # model.add(Dropout(KERAS_DROPOUT_RATE, seed=random_state))
        model.add(Dense(1, kernel_initializer='normal'))

        # Compile model
        optimizer = Adam(lr=KERAS_LEARNING_RATE, decay=decay)
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        # model = KerasRegressor(build_fn=nn,
        #                        nb_epoch=100, batch_size=5, verbose=VERBOSE)
        return model

    # return model based on model_choice
    def build_model(self=None, model_choice=XGB, random_state=1114):
        if model_choice == XGB:
            return self.xgb_model(random_state=random_state)
        elif model_choice == LIGHTGBM:
            return self.lgbm_model(random_state=random_state)
        elif model_choice == CATBOOST:
            return self.catboost_model(random_state=random_state)
        elif model_choice == ETREE:
            return self.etree_model(random_state=random_state)
        elif model_choice == RTREE:
            return self.rtree_model(random_state=random_state)
        elif model_choice == DTREE:
            return self.dtree_model(random_state=random_state)
        elif model_choice == VW:
            return self.vowpalwabbit_model()
        elif model_choice == KERAS:
            return self.keras_model(random_state=random_state)
        else:
            logger.error("Undefined model choice:" + str(model_choice))
            raise ValueError

    @timecall
    # @profile
    def train_model(self):
        model_choice = self.model_choice
        logger.info("Prepare data to train model")
        data = self.train_data
        target = data[self.label].values
        target_log = np.log(target)
        train_set = data.drop(
            ['id', self.label], axis=1).astype(float)
        # categorical_features_indices = self.convert_to_categrorical_features(
        #     train_set)
        logger.info("Training model ...." + str(model_choice))
        features = train_set.columns.values.tolist()
        logger.debug("Features:" + str(len(features)))
        logger.debug(features)
        cat_features = ['vendor_id', 'store_and_fwd_flag', 'pickup_month',
                        'pickup_weekday', 'pickup_day', 'pickup_hour',
                        'pickup_whour', 'pickup_minute'
                        ]
        logger.debug("Categorial features:")
        logger.debug(cat_features)
        logger.debug("Training ...")
        # logger.debug(train_set[cat_features].describe())
        X_train, X_test, Y_train, Y_test = train_test_split(
            train_set.values, target_log, train_size=0.85, random_state=1234)
        early_stopping_rounds = 50
        start = time.time()
        self.model = self.build_model(model_choice)
        model_name = self.model.__class__.__name__
        self.scaler = StandardScaler()
        if model_choice == VW:
            self.model.fit(X_train, Y_train)
        elif model_choice == CATBOOST:
            self.model.fit(
                X_train, Y_train, eval_set=(X_test, Y_test),
                use_best_model=True,
                verbose=VERBOSE
            )
        elif model_choice == LIGHTGBM:
            self.model.fit(
                X_train, Y_train, eval_set=[(X_test, Y_test)],
                eval_metric="rmse",
                early_stopping_rounds=early_stopping_rounds,
                verbose=VERBOSE,
                # feature_name=features,
                # categorical_feature=cat_features
                # categorical_feature=categorical_features_indices
            )
        elif model_choice == XGB:
            self.model.fit(
                X_train, Y_train, eval_set=[(X_test, Y_test)],
                eval_metric="rmse",
                early_stopping_rounds=early_stopping_rounds,
                # verbose=early_stopping_rounds
                verbose=VERBOSE
            )
        elif model_choice == KERAS:
            self.scaler.fit(X_train)
            X_train = self.scaler.transform(X_train)
            X_test = self.scaler.transform(X_test)
            self.model.fit(X_train, Y_train,
                           validation_data=(X_test, Y_test),
                           batch_size=KERAS_BATCH_SIZE,
                           epochs=KERAS_N_ROUNDS,
                           verbose=VERBOSE
                           )
        else:
            # model_choice == ETREE or model_choice == DTREE or
            self.model.fit(X_train, Y_train)

        end = time.time() - start
        logger.debug("Done training for " + model_name + ". Time:" + str(end))
        if model_choice == LIGHTGBM:
            logger.debug("Predicting for:" + model_name +
                         ". Best round:" + str(self.model.best_iteration))
            y_pred = self.model.predict(
                X_test, num_iteration=self.model.best_iteration)
        elif model_choice == XGB:
            logger.debug("Predicting for:" + model_name +
                         ". Best round:" + str(self.model.best_iteration) +
                         ". N_tree_limit:" + str(self.model.best_ntree_limit))
            y_pred = self.model.predict(
                X_test, ntree_limit=self.model.best_ntree_limit)
        elif model_choice == CATBOOST:
            logger.debug("Predicting for:" + model_name +
                         ". N_tree_limit:" + str(self.model.tree_count_))
            y_pred = self.model.predict(
                X_test, ntree_limit=self.model.tree_count_, verbose=VERBOSE)
        else:
            logger.debug("Predicting for:" + model_name)
            y_pred = self.model.predict(X_test)
        # score = self.rmsle(Y_test.values, y_pred, log=True)
        score1 = self.rmsle(Y_test, y_pred, log=False)
        logger.debug("RMSLE:" + str(score1))

    # Train using Kflow
    @timecall
    def train_kfold_single(self):
        logger.info("Prepare data to train model")
        data = self.train_data
        target = data[self.label]
        target_log = np.log(target)
        train_set = data.drop(
            ['id', self.label], axis=1).astype(float)
        total_rmse = 0
        kfolds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=321)
        self.model = XGBRegressor(n_estimators=N_ROUNDS, max_depth=MAX_DEPTH,
                                  learning_rate=LEARNING_RATE,
                                  min_child_weight=MIN_CHILD_WEIGHT, n_jobs=-1)
        X = train_set
        Y = target_log.values
        early_stopping_rounds = 50
        total_time = 0
        for j, (train_idx, test_idx) in enumerate(kfolds.split(X)):
            logger.debug("Round:" + str(j + 1))
            start = time.time()
            X_train = X.iloc[train_idx]
            Y_train = Y[train_idx]
            X_test = X.iloc[test_idx]
            Y_test = Y[test_idx]
            self.model.fit(X_train, Y_train, eval_set=[(X_test, Y_test)],
                           eval_metric="rmse",
                           early_stopping_rounds=early_stopping_rounds,
                           # verbose=early_stopping_rounds
                           verbose=VERBOSE
                           )
            end = time.time() - start
            total_time = total_time + end
            logger.debug("Done training for round:" + str(j + 1) +
                         " time:" + str(end) + "/" + str(total_time))
            y_pred = self.model.predict(X_test)
            rmse1 = self.rmsle(Y_test, y_pred, log=False)
            total_rmse = total_rmse + rmse1
            logger.debug("rmsle:" + str(rmse1))
        logger.debug("Avg rmse:" + str(total_rmse / (j + 1)))
        logger.debug("Total training time:" + str(total_time))

    # Train using Kflow and arrgregate trained models
    @timecall
    def train_kfold_aggregate(self):
        logger.info("Prepare data to train model")
        data = self.train_data
        target = data[self.label]
        target_log = np.log(target)
        train_set = data.drop(
            ['id', self.label], axis=1).astype(float)
        total_rmse = 0
        kfolds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=321)
        model = XGBRegressor(n_estimators=N_ROUNDS, max_depth=MAX_DEPTH,
                             learning_rate=LEARNING_RATE,
                             min_child_weight=MIN_CHILD_WEIGHT, n_jobs=-1)
        X = train_set
        Y = target_log.values
        T = self.eval_data.drop(
            ['id'], axis=1).astype(float)
        S_train = np.zeros((X.shape[0], N_FOLDS))
        S_test = np.zeros((T.shape[0], N_FOLDS))
        early_stopping_rounds = 50
        total_time = 0
        for j, (train_idx, test_idx) in enumerate(kfolds.split(X)):
            logger.debug("Round:" + str(j + 1))
            start = time.time()
            X_train = X.iloc[train_idx]
            Y_train = Y[train_idx]
            X_test = X.iloc[test_idx]
            Y_test = Y[test_idx]
            model.fit(X_train, Y_train, eval_set=[(X_test, Y_test)],
                      eval_metric="rmse",
                      early_stopping_rounds=early_stopping_rounds,
                      #   verbose=early_stopping_rounds
                      verbose=VERBOSE
                      )
            end = time.time() - start
            total_time = total_time + end
            logger.debug("Done training for round:" + str(j + 1) +
                         " time:" + str(end) + "/" + str(total_time))
            y_pred = model.predict(X_test)
            rmse1 = self.rmsle(Y_test, y_pred, log=False)
            total_rmse = total_rmse + rmse1
            logger.debug("rmsle:" + str(rmse1))
            logger.debug("Saving Y_pred for round:" + str(j + 1))
            S_train[:, j] = model.predict(X)
            logger.debug("Saving Y_eval for round:" + str(j + 1))
            S_test[:, j] = model.predict(T)
        logger.debug("Avg rmse:" + str(total_rmse / (j + 1)))
        logger.debug("Total training time:" + str(total_time))
        # logger.debug(S_train[:5])
        return S_train, S_test

    # Stack train from kfold
    @timecall
    def train_predict_kfold_aggregate(self):
        logger.info("Training model using kfold ang aggregate results")
        train_set, test_set = self.train_kfold_aggregate()
        target = self.train_data[self.label]
        target_log = np.log(target)

        X_train, X_test, Y_train, Y_test = train_test_split(
            train_set, target_log, train_size=0.85, random_state=1234)
        self.model = XGBRegressor(n_estimators=N_ROUNDS, max_depth=MAX_DEPTH,
                                  learning_rate=LEARNING_RATE,
                                  min_child_weight=MIN_CHILD_WEIGHT, n_jobs=-1)
        logger.debug("Training stack model ....")
        start = time.time()
        early_stopping_rounds = 50
        self.model.fit(
            X_train, Y_train, eval_set=[(X_train, Y_train), (X_test, Y_test)],
            eval_metric="rmse", early_stopping_rounds=early_stopping_rounds,
            # verbose=early_stopping_rounds
            verbose=VERBOSE
        )
        end = time.time() - start
        logger.debug("Done training:", end)
        y_pred = self.model.predict(X_test)
        score = self.rmsle(Y_test.values, y_pred, log=True)
        score1 = self.rmsle(Y_test.values, y_pred, log=False)
        logger.debug("RMSLE score:" + str(score) +
                     " RMSLE without-log:" + str(score1))
        logger.debug("Predict for eval set ..")
        Y_eval_log = self.model.predict(test_set)
        Y_eval = np.exp(Y_eval_log.ravel())
        logger.debug("Saving prediction to disk")
        eval_output = pd.DataFrame(
            {'id': self.eval_data['id'], self.label: Y_eval}, columns=['id', self.label])
        logger.debug("Eval data:" + str(len(eval_output)))
        today = str(dtime.date.today())
        eval_output.to_csv(
            DATA_DIR + '/' + today + '-submission.csv.gz', index=False,
            compression='gzip')

    @timecall
    def predict_save(self):
        model_choice = self.model_choice
        logger.info("Predicting for eval data ..")
        data = self.eval_data.drop('id', axis=1).astype(float).values
        if model_choice == LIGHTGBM:
            logger.debug("Predicting for:" + str(model_choice) +
                         ". Best round:" + str(self.model.best_iteration))
            Y_eval_log = self.model.predict(
                data, num_iteration=self.model.best_iteration)
        elif model_choice == XGB:
            logger.debug("Predicting for:" + str(model_choice) +
                         ". Best round:" + str(self.model.best_iteration) +
                         ". N_tree_limit:" + str(self.model.best_ntree_limit))
            Y_eval_log = self.model.predict(
                data, ntree_limit=self.model.best_ntree_limit)
        elif model_choice == KERAS:
            data = self.scaler.transform(data)
            Y_eval_log = self.model.predict(data)
        else:
            Y_eval_log = self.model.predict(data)
        Y_eval = np.exp(Y_eval_log.ravel())
        eval_output = self.eval_data.copy()
        eval_output.loc[:, self.label] = Y_eval
        logger.debug("Saving prediction to disk")
        today = str(dtime.date.today())
        eval_output[['id', self.label]].to_csv(
            DATA_DIR + '/' + today + '-submission.csv.gz', index=False,
            compression='gzip')

    def importance_features(self):
        logger.info("Feature importance:")
        threshold = 0
        data = self.train_data
        target = data[self.label]
        target_log = np.log(target)
        train_set = data.drop(
            ['id', self.label], axis=1).astype(float)
        features_score = pd.Series(
            self.model.feature_importances_, index=train_set.columns.values)
        # logger.debug("Feature importance:", features_score.describe())
        logger.debug(features_score.sort_values())
        return

    def plot_ft_importance(self):
        # logger.debug(self.model.feature_importances_)
        self.importance_features()
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111)
        plot_importance(self.model, ax=ax)
        plt.savefig(DATA_DIR + '/feature_importance.png')
        graph = xgb.to_graphviz(self.model)
        graph.render()

    # Convert input to vowpal_wabbit format
    @timecall
    def convert_2_vowpal_wabbit(self):
        logger.info("Converting train_pre to vowpal_wabbit")
        data = self.train_data
        target = data[self.label]
        target_log = np.log(target)
        train_set = data.drop(
            ['id', self.label], axis=1).astype(float)
        # X_train, X_test, Y_train, Y_test = train_test_split(
        #     train_set, target_log, train_size=0.85, random_state=1234)
        # vw_train = tovw(x=X_train, y=Y_train)
        vw_train = tovw(x=train_set, y=target_log)
        logger.debug("Saving train_vw ....")
        vw_train_pd = pd.Series(vw_train)
        train_file = DATA_DIR + "/train_vw.csv"
        vw_train_pd.to_csv(train_file, index=False)
        # logger.debug("Saving cv_vw ....")
        # cv_file = DATA_DIR + "/cv_vw.csv"
        # vw_cv = tovw(x=X_test, y=None)
        # vw_cv_pd = pd.Series(vw_cv)
        # vw_cv_pd.to_csv(cv_file, index=False)
        # self.train_vowpal_wabbit_from_file()
        # self.score_vowpal_wabbit(Y_test.values)

        logger.debug("Converting test_pre to vowpal_wabbit")
        eval_file = DATA_DIR + "/test_vw.csv"
        data = self.eval_data.drop('id', axis=1).astype(float)
        vw_eval = tovw(x=data, y=None)
        logger.debug("Saving eval_vw ....")
        vw_eval_pd = pd.Series(vw_eval)
        vw_eval_pd.to_csv(eval_file, index=False)
        logger.debug("Done save_vw")

    # Train model using vowpal_wabbit
    @timecall
    def train_vowpal_wabbit(self):
        logger.info("Training model using vowpal_wabbit")
        cache_file = DATA_DIR + "/train_vw.cache"
        try:
            os.remove(cache_file)
        except:
            pass
        train_file = DATA_DIR + "/train_vw.csv"
        model_file = DATA_DIR + "/model.vw"
        num_passes = 1000
        learning_rate = 0.1
        command = "/usr/local/bin/vw " + train_file + " --cache_file " + \
            cache_file + " --passes " + str(num_passes) + " -f " + model_file + \
            " --noconstant" + " --learning_rate " + str(learning_rate)
        # + " -q ::"
        logger.debug(command)
        # result = subprocess.check_output(command, stderr=subprocess.STDOUT,
        # shell=True)
        result = subprocess.call(command, stderr=subprocess.STDOUT, shell=True)

    # Score model using vowpal_wabbit
    @timecall
    def score_vowpal_wabbit(self, Y_test):
        logger.info("Scoring model using vowpal_wabbit")
        cv_file = DATA_DIR + "/cv_vw.csv"
        model_file = DATA_DIR + "/model.vw"
        pred_file = DATA_DIR + "/cv_predict.csv"
        command = "/usr/local/bin/vw -t " + cv_file + \
            " -i " + model_file + " -p " + pred_file + " --quiet"
        logger.debug("Predicting cross validation model using vowpal_wabbit")
        logger.debug(command)
        result = subprocess.check_output(
            command, stderr=subprocess.STDOUT, shell=True)
        pred_pd = pd.read_csv(pred_file, header=None)
        Y_pred_log = pred_pd.values
        score = self.rmsle(y=Y_test, y_pred=Y_pred_log, log=False)
        logger.debug("RMLSE score:" + str(score))

    # Predict model using vowpal_wabbit
    @timecall
    def predict_vowpal_wabbit(self):
        eval_file = DATA_DIR + "/test_vw.csv"
        model_file = DATA_DIR + "/model.vw"
        pred_file = DATA_DIR + "/test_predict.csv"
        command = "/usr/local/bin/vw -t " + eval_file + \
            " -i " + model_file + " -p " + pred_file + " --quiet"
        logger.debug("Predicting for eval model ..")
        logger.debug(command)
        result = subprocess.check_output(
            command, stderr=subprocess.STDOUT, shell=True)
        # result = subprocess.call(command, stderr=subprocess.STDOUT,
        # shell=True)

    # predict and save trip_durarion based on vowpal_wabbit
    @timecall
    def predict_save_vowpal_wabbit(self):
        self.predict_vowpal_wabbit()
        pred_file = DATA_DIR + "/test_predict.csv"
        data = self.eval_data.drop('id', axis=1).astype(float)
        pred_pd = pd.read_csv(pred_file, header=None)
        Y_eval_log = pred_pd.values
        Y_eval = np.exp(Y_eval_log.ravel())
        eval_output = self.eval_data.copy()
        eval_output.loc[:, self.label] = Y_eval
        logger.debug("Saving prediction to disk")
        today = str(dtime.date.today())
        eval_output[['id', self.label]].to_csv(
            DATA_DIR + '/' + today + '-submission.csv.gz', index=False,
            compression='gzip')

    # Kfold, train for single model in stack
    @timecall
    # @profile => Do not enable because of bug in CatBoost
    def train_base_single_model(self, model_index, model_choice):
        logger.info("Prepare data to train base model")
        data = self.train_data
        target = data[self.label]
        target_log = np.log(target)
        train_set = data.drop(
            ['id', self.label], axis=1).astype(float)
        total_rmse = 0
        X_in = train_set.values
        Y_in = target_log.values
        T_in = self.eval_data.drop(
            ['id'], axis=1).astype(float).values
        logger.debug("X shape:" + str(X_in.shape) + " Y shape:" +
                     str(Y_in.shape) + " Test shape:" + str(T_in.shape))
        n_folds = N_FOLDS
        kfolds = KFold(n_splits=n_folds, shuffle=True, random_state=321)
        S_train = np.zeros(X_in.shape[0])
        S_test = np.zeros(T_in.shape[0])
        S_test_i = np.zeros((T_in.shape[0], n_folds))
        model = self.build_model(model_choice)
        model_name = model.__class__.__name__
        logger.debug("Base model " + model_name)
        logger.debug("S_train shape:" + str(S_train.shape) + " S_test shape:" +
                     str(S_test.shape))
        early_stopping_rounds = 50
        model_rmse = 0
        start_sub = time.time()
        for j, (train_idx, test_idx) in enumerate(kfolds.split(X_in)):
            logger.debug("fold:" + str(j + 1) + " begin training ...")
            X_train = np.copy(X_in[train_idx])
            Y_train = np.copy(Y_in[train_idx])
            X_holdout = np.copy(X_in[test_idx])
            y_holdout = np.copy(Y_in[test_idx])
            if model_name == 'XGBRegressor':
                model.fit(
                    X_train, Y_train, eval_set=[(X_holdout, y_holdout)],
                    eval_metric="rmse",
                    early_stopping_rounds=early_stopping_rounds,
                    # verbose=10
                    verbose=VERBOSE
                )
                logger.debug("fold:" + str(j + 1) +
                             " done training. Best round:" +
                             str(model.best_ntree_limit) + " . Predicting for RMSLE")
                y_pred = model.predict(
                    X_holdout, ntree_limit=model.best_ntree_limit)[:]
                S_train[test_idx] = y_pred
                S_test_i[:, j] = model.predict(
                    T_in, ntree_limit=model.best_ntree_limit)[:]
            elif model_name == 'LGBMRegressor':
                model.fit(
                    X_train, Y_train, eval_set=[(X_holdout, y_holdout)],
                    eval_metric="rmse",
                    early_stopping_rounds=early_stopping_rounds,
                    # verbose=10,
                    verbose=VERBOSE
                    # feature_name=features,
                    # categorical_feature=cat_features
                    # categorical_feature=categorical_features_indices
                )
                logger.debug("fold:" + str(j + 1) +
                             " done training. Best round:" +
                             str(model.best_iteration) + " . Predicting for RMSLE")
                y_pred = model.predict(
                    X_holdout, num_iteration=model.best_iteration)[:]
                S_train[test_idx] = y_pred
                S_test_i[:, j] = model.predict(
                    T_in, num_iteration=model.best_iteration)[:]
            elif model_name == "CatBoostRegressor":
                model.fit(
                    X_train, Y_train, eval_set=(X_holdout, y_holdout),
                    use_best_model=True,
                    verbose=VERBOSE
                )
                logger.debug("fold:" + str(j + 1) +
                             " done training. Best round:" +
                             str(model.tree_count_) + " . Predicting for RMSLE")
                y_pred = model.predict(
                    X_holdout, ntree_limit=model.tree_count_, verbose=VERBOSE)
                S_train[test_idx] = y_pred
                S_test_i[:, j] = model.predict(
                    T_in, ntree_limit=model.tree_count_)[:]
            elif model_choice == KERAS:
                self.scaler = StandardScaler()
                self.scaler.fit(X_train)
                X_train = self.scaler.transform(X_train)
                X_holdout = self.scaler.transform(X_holdout)
                model.fit(X_train, Y_train,
                          validation_data=(
                              X_holdout, y_holdout),
                          batch_size=KERAS_BATCH_SIZE,
                          epochs=KERAS_N_ROUNDS,
                          verbose=VERBOSE
                          )
                logger.debug("fold:" + str(j + 1) +
                             " done training. Predicting for RMSLE")
                y_pred = np.ravel(model.predict(X_holdout)[:])
                print(S_train.shape, y_pred.shape)
                S_train[test_idx] = y_pred
                T_in_temp = self.scaler.transform(T_in)
                S_test_i[:, j] = np.ravel(model.predict(T_in_temp)[:])
            else:
                model.fit(X_train, Y_train)
                logger.debug("fold:" + str(j + 1) +
                             " done training. Predicting for RMSLE")
                y_pred = model.predict(X_holdout)[:]
                S_train[test_idx] = y_pred
                S_test_i[:, j] = model.predict(T_in)[:]

            rmse1 = self.rmsle(y_holdout, y_pred, log=False)
            model_rmse = model_rmse + rmse1
            logger.debug("fold:" + str(j + 1) + " rmse:" + str(rmse1))
            # end of for j

        S_test = S_test_i.mean(1)
        logger.debug("Saving trained model data ...")
        train_file, test_file = self.save_trained_single_model_data(
            model_index, model_name, S_train, S_test)
        end_sub = time.time() - start_sub
        logger.debug("Model rmse:" + str(model_rmse / (j + 1)))
        logger.debug("Done training for " + model_name +
                     ". Trained time:" + str(end_sub))
        # cleanup memory
        # del S_train
        # del S_test
        # del S_test_i
        # del model
        # gc.collect()
        # end of for i
        return model_rmse, train_file, test_file

    # save pretrained data from train_base_single_model
    @timecall
    def save_trained_single_model_data(self, model_index, model_name, X_in, T_in):
        logger.info("Saving pretrained data .. ")
        col = str(model_index)
        d_train = pd.DataFrame(data=X_in, columns=[col])
        d_train['id'] = self.train_data['id']
        train_file = DATA_DIR + '/train_base_' + \
            model_name + '.csv'
        d_train.to_csv(train_file, columns=['id', col], index=False)
        d_test = pd.DataFrame(data=T_in, columns=[col])
        d_test['id'] = self.eval_data['id']
        test_file = DATA_DIR + '/test_base_' + \
            model_name + '.csv'
        d_test.to_csv(test_file, columns=['id', col], index=False)
        # cleanup memory
        # del d_train
        # del d_test
        return train_file, test_file

    # Kfold, train for each model, stack result
    @timecall
    # @profile  => Do not enable because of bug in CatBoost
    def train_base_models(self):
        # Credit
        # to:https://dnc1994.com/2016/05/rank-10-percent-in-first-kaggle-competition-en/
        logger.info("Training for model level 1 ...")
        # models = [RTREE, ETREE, CATBOOST, XGB, LIGHTGBM]
        # models = [ETREE, CATBOOST, XGB, LIGHTGBM]
        # models = [CATBOOST, XGB, LIGHTGBM]
        models = [XGB, LIGHTGBM]
        # models = [RTREE, ETREE, CATBOOST]
        all_rmse = 0
        start = time.time()
        train_files = []
        test_files = []
        start = time.time()
        for i, model_choice in enumerate(models):
            # model = models[i]
            mem = str(psutil.virtual_memory())
            logger.debug("Memmory before training single model:")
            logger.debug("Before: " + mem)
            model_rmse, train_file, test_file = self.train_base_single_model(
                i, model_choice)
            mem = str(psutil.virtual_memory())
            logger.debug("Memmory after training single model:")
            logger.debug("After: " + mem)
            all_rmse = all_rmse + model_rmse
            train_files.append(train_file)
            test_files.append(test_file)

        logger.debug("Saving all trained model datas ... ")
        self.save_trained_models_data(train_files, test_files)
        end = time.time() - start
        logger.debug("All AVG rmse:" + str(all_rmse / len(models)))
        logger.info("Done training base models:" + str(end))

    # combine all trained model datas into 1 train_stack and test_stack
    @timecall
    def save_trained_models_data(self, train_files, test_files):
        logger.info("Saving all trained data to combined one ..")
        data = self.train_data
        target = data[self.label]
        target_log = np.log(target)
        train_datas = []
        begin = True
        for i, train_file in enumerate(train_files):
            train_data = pd.read_csv(train_file)
            if begin:
                total_train_data = train_data
                begin = False
            else:
                total_train_data = total_train_data.join(
                    train_data.set_index('id'), on='id', rsuffix=str(i))
        logger.debug("Join base data with pre data")
        pre_data = pd.DataFrame(
            {'id': data['id'], self.label: target_log}, columns=['id', self.label])
        # total_train_data.loc[:, self.label] = target_log[:len(total_train_data)]
        total_train_data = total_train_data.join(
            pre_data.set_index('id'), on='id')
        # Drop NaN row incase of each data set has different rows
        total_train_data = total_train_data.dropna()
        total_train_data.to_csv(DATA_DIR + '/train_stack.csv', index=False)
        logger.debug("Train stack size:" + str(total_train_data.shape))
        begin = True
        for i, test_file in enumerate(test_files):
            test_data = pd.read_csv(test_file)
            if begin:
                total_test_data = test_data
                begin = False
            else:
                total_test_data = total_test_data.join(
                    test_data.set_index('id'), on='id', rsuffix=str(i))
        total_test_data.to_csv(DATA_DIR + '/test_stack.csv', index=False)
        logger.debug("Test stack size:" + str(total_test_data.shape))

    # combine all pretrained data from single base model
    def combine_pretrained_data(self):
        logger.info("Combine pretrained data from single base models")
        train_path = DATA_DIR + "/train_base*.csv"
        train_filenames = glob.glob(train_path)
        test_path = DATA_DIR + "/test_base*.csv"
        test_filenames = glob.glob(test_path)
        logger.debug("Number of train base files:" + str(len(train_filenames)))
        logger.debug(', '.join(train_filenames))
        logger.debug("Number of test base files:" + str(len(test_filenames)))
        logger.debug(', '.join(test_filenames))
        self.save_trained_models_data(train_filenames, test_filenames)

    # load pretrained data
    @timecall
    def load_pretrained_data(self):
        logger.info("Loading pretrained data .. ")
        d_train = pd.read_csv(DATA_DIR + '/train_stack.csv')
        d_test = pd.read_csv(DATA_DIR + '/test_stack.csv')
        return d_train, d_test

    # Train stack using Kflow and arrgregate predicted datas
    @timecall
    # @profile
    def train_stack_model_kfold(self, model_choice=XGB):
        logger.info("Prepare data to train stack model using kfold")
        S_train, S_test = self.load_pretrained_data()
        logger.debug("S_train shape:" + str(S_train.shape) + " S_test shape:" +
                     str(S_test.shape))
        target_log = S_train[self.label]
        train_set = S_train.drop([self.label, 'id'], axis=1)
        total_rmse = 0
        early_stopping_rounds = 10
        n_folds = N_FOLDS
        kfolds = KFold(n_splits=n_folds, shuffle=True, random_state=321)
        X_in = train_set.values
        Y_in = target_log.values
        T_in = S_test.drop(['id'], axis=1).values
        S_test_total = np.zeros((T_in.shape[0], 1))
        S_test_kfold = np.zeros((T_in.shape[0], n_folds))
        early_stopping_rounds = 10
        total_time = 0

        if model_choice == LIGHTGBM:
            model = LGBMRegressor(objective='regression_l2',
                                  metric='l2_root',
                                  n_estimators=N_ROUNDS,
                                  #   n_estimators=10,
                                  learning_rate=0.01,
                                  num_leaves=1024,
                                  #  max_bin=1024,
                                  #  min_data_in_leaf=100,
                                  seed=1111,
                                  nthread=-1, silent=SILENT)

        elif model_choice == XGB:
            model = XGBRegressor(
                n_estimators=N_ROUNDS,
                # n_estimators=10,
                max_depth=10,
                learning_rate=0.03,
                min_child_weight=1,
                #  gamma=0,
                random_state=567,
                n_jobs=-1,
                silent=SILENT
            )
        model_name = model_name = model.__class__.__name__
        start = time.time()
        for j, (train_idx, test_idx) in enumerate(kfolds.split(X_in)):
            logger.debug("fold:" + str(j + 1) + " begin training ...")
            X_train = X_in[train_idx]
            Y_train = Y_in[train_idx]
            X_holdout = X_in[test_idx]
            y_holdout = Y_in[test_idx]
            if model_name == 'XGBRegressor':
                model.fit(
                    X_train, Y_train, eval_set=[(X_holdout, y_holdout)],
                    eval_metric="rmse",
                    early_stopping_rounds=early_stopping_rounds,
                    # verbose=10
                    verbose=VERBOSE
                )
                logger.debug("fold:" + str(j + 1) +
                             " done training. Best round:" +
                             str(model.best_ntree_limit) + " . Predicting for RMSLE")
                y_pred = model.predict(
                    X_holdout, ntree_limit=model.best_ntree_limit)[:]
                S_test_kfold[:, j] = model.predict(
                    T_in, ntree_limit=model.best_ntree_limit)[:]
            elif model_name == 'LGBMRegressor':
                model.fit(
                    X_train, Y_train, eval_set=[(X_holdout, y_holdout)],
                    eval_metric="rmse",
                    early_stopping_rounds=early_stopping_rounds,
                    # verbose=10,
                    verbose=VERBOSE
                    # feature_name=features,
                    # categorical_feature=cat_features
                    # categorical_feature=categorical_features_indices
                )
                logger.debug("fold:" + str(j + 1) +
                             " done training. Best round:" +
                             str(model.best_iteration) + " . Predicting for RMSLE")
                y_pred = model.predict(
                    X_holdout, num_iteration=model.best_iteration)[:]
                S_test_kfold[:, j] = model.predict(
                    T_in, num_iteration=model.best_iteration)[:]
            elif model_name == "CatBoostRegressor":
                model.fit(
                    X_train, Y_train, eval_set=(X_holdout, y_holdout),
                    use_best_model=True,
                    verbose=VERBOSE
                )
                logger.debug("fold:" + str(j + 1) +
                             " done training. Best round:" +
                             str(model.tree_count_) + " . Predicting for RMSLE")
                y_pred = model.predict(
                    X_holdout, ntree_limit=model.tree_count_, verbose=VERBOSE)
                S_test_kfold[:, j] = model.predict(
                    T_in, ntree_limit=model.tree_count_)[:]

            rmse1 = self.rmsle(y_holdout, y_pred, log=False)
            total_rmse = total_rmse + rmse1
            logger.debug("fold:" + str(j + 1) + " rmse:" + str(rmse1))
            # end of for j

        S_test_total = S_test_kfold.mean(1)
        end = time.time() - start
        logger.debug("All rmse:" + str(total_rmse / (j + 1)))
        logger.debug("Done training for " + model_name +
                     ". Trained time:" + str(end))
        self.save_stacked_data(S_test_total)

    @timecall
    def save_stacked_data(self, Y_eval_log):
        logger.info("Saving submission for stack model to disk")
        Y_eval = np.exp(Y_eval_log.ravel())
        # print(Y_eval_log[:5])
        # print(Y_eval[:5])
        eval_output = self.eval_data
        logger.debug("Size of eval data:" + str(len(eval_output)
                                                ) + " . Size of Y_eval:" + str(len(Y_eval)))
        eval_output.loc[:, self.label] = Y_eval
        today = str(dtime.date.today())
        eval_output[['id', self.label]].to_csv(
            DATA_DIR + '/' + today + '-submission.csv.gz', index=False,
            compression='gzip')

    # Cross validation for xgboost model
    @timecall
    def xgb_cv(self):
        logger.info("Prepare data to CV XGBoost model")
        data = self.train_data
        target = data[self.label]
        target_log = np.log(target)
        train_set = data.drop(
            ['id', self.label], axis=1).astype(float)
        # X_train, X_test, Y_train, Y_test = train_test_split(
        #     train_set, target_log, train_size=0.85, random_state=1234)
        # lgb_train = lgb.Dataset(X_test, Y_test)
        lgb_train = xgb.DMatrix(train_set, target_log)
        params = {
            'learning_rate': 0.1,
            'max_depth': 10,
            'min_child_weight': 1,
            # 'gamma': 1,
            # 'silent': 1
        }
        early_stopping_rounds = 10
        cv_results = xgb.cv(params, lgb_train, num_boost_round=200, nfold=5,
                            metrics="rmse", shuffle=True,
                            early_stopping_rounds=early_stopping_rounds,
                            verbose_eval=10, show_stdv=True, seed=1000)

    # Cross validation for lightgbm model
    @timecall
    def lgbm_cv(self):
        logger.info("Prepare data to CV LightGBM model")
        data = self.train_data
        target = data[self.label]
        target_log = np.log(target)
        train_set = data.drop(
            ['id', self.label], axis=1).astype(float)
        # X_train, X_test, Y_train, Y_test = train_test_split(
        #     train_set, target_log, train_size=0.85, random_state=1234)
        # lgb_train = lgb.Dataset(X_test, Y_test)
        lgb_train = lgb.Dataset(train_set, target_log)
        params = {
            'objective': 'regression_l2',
            'metric': 'l2_root',
            'learning_rate': 0.1,
            'num_leaves': 1024,
            # 'max_bin': 1024,
            # 'min_data_in_leaf': 100,
            # 'nthread': -1,
            'verbose': 1
        }
        early_stopping_rounds = 10
        cv_results = lgb.cv(params, lgb_train, num_boost_round=300, nfold=5,
                            metrics="rmse", shuffle=True,
                            early_stopping_rounds=early_stopping_rounds,
                            verbose_eval=10, show_stdv=True, seed=1000)
        # logger.debug(cv_results)
        # logger.debug('Best num_boost_round:', len(cv_results['l1-mean']))
        # logger.debug('Best CV score:', cv_results['l1-mean'][-1])


# ---------------- Main -------------------------
if __name__ == "__main__":
    start = time.time()
    option = 2
    model_choice = KERAS
    logger = logging.getLogger('newyork-taxi-duration')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(DATA_DIR + '/model.log', mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.info("Runing on " + os.name)
    base_class = TaxiTripDuration(LABEL, model_choice)
    # Load and preprocessed data
    if option == 1:
        base_class.load_data()
        base_class.check_null_data()
        base_class.preprocess_data()
        base_class.check_null_data()
        base_class.feature_correlation()

    # ------------------------ single model -----------------------
    # Load process data and train model
    elif option == 2:
        base_class.load_preprocessed_data()
        base_class.train_model()
        base_class.predict_save()
        # base_class.importance_features()
        # base_class.plot_ft_importance()
    # Load process data and train model with Kfold
    elif option == 3:
        base_class.load_preprocessed_data()
        base_class.train_kfold_single()
        base_class.predict_save()
        # base_class.importance_features()
        # base_class.plot_ft_importance()
    # Load process data and train model with Kfold, aggregate result then
    # train again with aggregated data
    # Note: LB lower than Kfold single
    elif option == 4:
        base_class.load_preprocessed_data()
        base_class.train_predict_kfold_aggregate()

    # ------------------------ stacking model -----------------------
    # Load process data and train all base models.
    # After that combine all single model datas in one set
    elif option == 5:
        base_class.load_preprocessed_data()
        base_class.train_base_models()

    # Load pretrained data and stacking model using kfold
    elif option == 6:
        base_class.load_preprocessed_data()
        base_class.train_stack_model_kfold(model_choice=XGB)

    # Load process data and train all base models. Alter that stacking model
    # using kfold
    elif option == 7:
        base_class.load_preprocessed_data()
        base_class.train_base_models()
        base_class.train_stack_model_kfold(model_choice=XGB)

    # Combine all previous pretrained datas and stacking model using kfold
    elif option == 8:
        base_class.load_preprocessed_data()
        base_class.combine_pretrained_data()
        base_class.train_stack_model_kfold(model_choice=XGB)

    # Train for single base model, combine with previous pretrained data and
    # stacking model using kfold
    elif option == 9:
        base_class.load_preprocessed_data()
        base_class.train_base_single_model(model_index=6, model_choice=KERAS)
        base_class.combine_pretrained_data()
        base_class.train_stack_model_kfold(model_choice=XGB)

    # ------------------- Other ------------------------------
    # Load process data and search for best model parameters
    elif option == 19:
        base_class.load_preprocessed_data()
        base_class.search_best_model_params()

    # ------------------ vowpal_wabbit---------------------------
    # Load process data and save to vowpal_wabbit format
    elif option == 21:
        base_class.load_preprocessed_data()
        base_class.convert_2_vowpal_wabbit()
    # Load pre-vw data and train using  vowpal_wabbi
    elif option == 22:
        base_class.load_preprocessed_data()
        base_class.train_vowpal_wabbit()
        base_class.predict_save_vowpal_wabbit()
    # convert to vw, train and predict model using vowpal_wabbi
    elif option == 23:
        base_class.load_preprocessed_data()
        base_class.convert_2_vowpal_wabbit()
        base_class.train_vowpal_wabbit()
        base_class.predict_save_vowpal_wabbit()

    # ----------------------LIGHTGBM ----------------------
    # cross validate lightgbm, to find best model hyper-parameters
    elif option == 33:
        base_class.load_preprocessed_data()
        base_class.lgbm_cv()

    # -------------------- XGBOOST --------------
    elif option == 43:
        base_class.load_preprocessed_data()
        base_class.xgb_cv()

    # ------------------------------ default -------------------------
    # combine preprocess and training model
    else:
        base_class.load_data()
        base_class.check_null_data()
        base_class.preprocess_data()
        base_class.check_null_data()
        base_class.feature_correlation()
        base_class.train_model()
        base_class.predict_save()
        # base_class.importance_features()
        # base_class.plot_ft_importance()
    end = time.time() - start
    logger.info("Done. Total time:" + str(end))
