# New York City Taxi Trip Duration
# Share code and data to improve ride time predictions
# https://www.kaggle.com/c/nyc-taxi-trip-duration/data
# install: http://jupyter-contrib-nbextensions.readthedocs.io/en/latest/install.html
# install: https://github.com/Jupyter-contrib/jupyter_nbextensions_configurator
# install GDAL: https://sandbox.idre.ucla.edu/sandbox/tutorials/installing-gdal-for-windows
# install osmnx: http://geoffboeing.com/2014/09/using-geopandas-windows/
# Credit to:
# https://www.kaggle.com/ankasor/driving-distance-using-open-street-maps-data/notebook
# https://www.kaggle.com/oscarleo/new-york-city-taxi-with-osrm
# https://www.kaggle.com/gaborfodor/from-eda-to-the-top-lb-0-368/notebook
# https://www.kaggle.com/headsortails/nyc-taxi-eda-update-the-fast-the-classified/notebook
# https://www.kaggle.com/mathijs/weather-data-in-new-york-city-2016

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
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score, train_test_split, learning_curve, validation_curve, KFold
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.cluster import MiniBatchKMeans, KMeans
# # XGB
from xgboost.sklearn import XGBRegressor
import xgboost as xgb
from xgboost import plot_importance
# # CatBoost
# from catboost import Pool, CatBoostRegressor
# Vowpal Wabbit
# from vowpalwabbit.sklearn_vw import VWRegressor
# from vowpalwabbit.sklearn_vw import tovw
# LightGBM
from lightgbm import LGBMRegressor
import lightgbm as lgb
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
# Other
# from geographiclib.geodesic import Geodesic
# import osmnx as ox
# import networkx as nx

pd.options.display.float_format = '{:,.4f}'.format
# Input data files are available in the DATA_DIR directory.
DATA_DIR = "data-temp"
LABEL = 'trip_duration'
N_FOLDS = 5
N_CLUSTERS = 200  # Kmeans number of cluster
# Model choice
XGB = 1
VW = 2
CATBOOST = 3
LIGHTGBM = 4
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
N_ROUNDS = 15000
# N_ROUNDS = 10
LOG_LEVEL = logging.DEBUG


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
        # Download from:
        # https://www.kaggle.com/oscarleo/new-york-city-taxi-with-osrm
        train_osm = pd.read_csv(DATA_DIR + "/fastest_route_train.csv")
        eval_osm = pd.read_csv(DATA_DIR + "/fastest_route_test.csv")
        logger.debug("train size:" + str(train_data.shape) +
                     " test size:" + str(eval_data.shape))
        logger.debug("train_osm size:" + str(train_osm.shape) +
                     " test osm size:" + str(eval_osm.shape))

        logger.debug("Merging  2 data sets ...")
        col_use = ['id', 'total_distance', 'total_travel_time',
                   'number_of_steps',
                   'starting_street', 'end_street', 'step_maneuvers', 'step_direction']
        train_osm_data = train_osm[col_use]
        eval_osm_data = eval_osm[col_use]
        train_data = train_osm_data.join(train_data.set_index('id'), on='id')
        # Cleanup data
        train_data = self.cleanup_data(train_data)
        eval_data = eval_osm_data.join(eval_data.set_index('id'), on='id')
        features = eval_data.columns.values
        self.target = train_data[label]
        self.combine_data = pd.concat(
            [train_data[features], eval_data], keys=['train', 'eval'])
        # self.load_and_combine_weather_data() => No score change
        logger.debug("combine data:" + str(len(self.combine_data)))
        features = self.combine_data.columns.values
        logger.debug("Original features:" + str(len(features)))
        logger.debug(features)
        logger.info("Data loaded")

    def load_and_combine_weather_data(self):
        logger.debug("Loading weather data ..")
        weather_data = pd.read_csv(
            DATA_DIR + "/weather_data_nyc_centralpark_2016.csv")
        logger.debug("Weather data len:", len(weather_data))
        # Convert date string to date_obj
        weather_data.loc[:, 'date_obj'] = pd.to_datetime(
            weather_data['date'], dayfirst=True).dt.date
        # convert object columns
        col = 'precipitation'
        T_value = 0.01
        weather_data.loc[weather_data[col] == 'T'] = T_value
        weather_data.loc[:, col] = weather_data[col].astype(float)

        col = 'snow fall'
        weather_data.loc[weather_data[col] == 'T'] = T_value
        weather_data.loc[:, col] = weather_data[col].astype(float)

        col = 'snow depth'
        weather_data.loc[weather_data[col] == 'T'] = T_value
        weather_data.loc[:, col] = weather_data[col].astype(float)
        weather_data.drop('date', axis=1, inplace=True)

        weather_data = weather_data.set_index('date_obj')

        # Convert combine_data datetime string to date_obj => join with weather
        self.combine_data.loc[:, 'date_obj'] = pd.to_datetime(
            self.combine_data['pickup_datetime'], format="%Y-%m-%d").dt.date
        # Join combine_data and weather_data
        self.combine_data = self.combine_data.join(
            weather_data, on='date_obj')

        # Drop un-used cols
        self.combine_data.drop('date_obj', axis=1, inplace=True)

    @timecall
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
        data = data[(data[label] < 22 * 3600) & (data[label] > 10)]
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
        train_set.loc[:, self.label] = self.target
        train_set.loc[(train_set[col] == 0.) & (
            train_set[col1] == 0.), self.label] = 0
        self.target = train_set[self.label]
        self.estimate_total_distance()

    def cal_speed(self, distance_col):
        col = 'speed'
        data_speed = self.combine_data.loc['train'].copy()
        data_speed.loc[:, self.label] = self.target
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
        data.loc[:, self.label] = self.target
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
        # logger.debug("Cluster for starting_street_tf")
        # col_use = ['pickup_hour', 'starting_street_tf']
        # col_cluster = 'pkhour_starting_street_cluster'
        # clusters = self.cal_cluster(data[col_use], N_CLUSTERS)
        # data.loc[:, col_cluster] = clusters
        logger.info("Cluster for starting_street_tf, end_street_tf")
        col_use = ['starting_street_tf', 'end_street_tf']
        col_cluster = 'location_cluster'
        clusters = self.cal_cluster(data[col_use], N_CLUSTERS)
        data.loc[:, col_cluster] = clusters
        logger.debug("Cluster for pickup")
        col_use = ['pickup_latitude', 'pickup_longitude']
        col_cluster = 'pickup_cluster'
        clusters = self.cal_cluster(data[col_use], N_CLUSTERS)
        data.loc[:, col_cluster] = clusters
        logger.debug("Cluster for dropoff")
        col_use = ['dropoff_latitude', 'dropoff_longitude']
        col_cluster = 'dropoff_cluster'
        clusters = self.cal_cluster(data[col_use], N_CLUSTERS)
        data.loc[:, col_cluster] = clusters

    def drop_unused_cols(self):
        data = self.combine_data
        data.drop(['pickup_datetime', 'datetime_obj', 'starting_street',
                   'end_street', 'step_maneuvers', 'step_direction', 'pickup_year'], axis=1, inplace=True)

    @timecall
    def preprocess_data(self):
        logger.info("Preproccesing data ...")
        self.fillnan()
        self.convert_datetime()
        self.convert_starting_street()
        self.convert_end_street()
        self.convert_store_and_fwd_flag()
        self.feature_haversine()
        self.feature_manhattan()
        self.feature_left_turns()
        self.feature_right_turns()
        self.feature_cluster()
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
        train_set.loc[:, self.label] = self.target
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
        data.loc[:, self.label] = self.target
        correlation = data.corr()[self.label].sort_values()
        logger.debug(str(correlation))

    def xgb_model(self, learning_rate=LEARNING_RATE, random_state=1000):
        # self.model = XGBRegressor(n_estimators=N_ROUNDS, max_depth=MAX_DEPTH,
        #                           learning_rate=LEARNING_RATE,
        #                           min_child_weight=MIN_CHILD_WEIGHT,
        #                           colsample_bytree=COLSAMPLE_BYTREE,
        #                           n_jobs=-1)
        # model = XGBRegressor(n_estimators=N_ROUNDS, max_depth=MAX_DEPTH,
        #                      learning_rate=LEARNING_RATE,
        #                      min_child_weight=MIN_CHILD_WEIGHT,
        #                      gamma=0,
        #                      random_state=1000,
        #                      n_jobs=-1)
        model = XGBRegressor(n_estimators=N_ROUNDS,
                             max_depth=10,
                             learning_rate=learning_rate,
                             min_child_weight=1,
                             #  gamma=0,
                             random_state=random_state,
                             n_jobs=-1,
                             silent=False
                             )
        return model

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
            'min_child_weight': 2,
            # 'gamma': 1,
            # 'silent': 1
        }
        early_stopping_rounds = 10
        cv_results = xgb.cv(params, lgb_train, num_boost_round=200, nfold=5,
                            metrics="rmse", shuffle=True,
                            early_stopping_rounds=early_stopping_rounds,
                            verbose_eval=10, show_stdv=True, seed=1000)

    def lgbm_model(self, random_state=1024):
        # model = LGBMRegressor(objective='regression_l2',
        #                       metric='l2_root',
        #                       n_estimators=N_ROUNDS,
        #                       #   n_estimators=10,
        #                       #   max_depth=MAX_DEPTH,
        #                       learning_rate=0.01,
        #                       #   min_child_weight=MIN_CHILD_WEIGHT,
        #                       num_leaves=4096,
        #                       max_bin=1024,
        #                       min_data_in_leaf=100,
        #                       seed=random_state,
        #                       nthread=-1, silent=False)
        model = LGBMRegressor(objective='regression_l2',
                              metric='l2_root',
                              n_estimators=N_ROUNDS,
                              #   n_estimators=10,
                              learning_rate=0.01,
                              num_leaves=1024,
                              seed=random_state,
                              nthread=-1, silent=False)
        return model

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
            'learning_rate': 0.01,
            'num_leaves': 4096,
            'max_bin': 1024,
            'min_data_in_leaf': 100,
            # 'nthread': -1,
            'verbose': 0
        }
        early_stopping_rounds = 50
        cv_results = lgb.cv(params, lgb_train, num_boost_round=300, nfold=5,
                            metrics="rmse", shuffle=True,
                            early_stopping_rounds=early_stopping_rounds,
                            verbose_eval=10, show_stdv=True, seed=1000)
        # logger.debug(cv_results)
        # logger.debug('Best num_boost_round:', len(cv_results['l1-mean']))
        # logger.debug('Best CV score:', cv_results['l1-mean'][-1])

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

    def catboost_model(self):
        model = CatBoostRegressor(
            iterations=N_ROUNDS * 3, learning_rate=0.1, depth=MAX_DEPTH,
            use_best_model=True, train_dir=DATA_DIR + "/node_modules",
            verbose=True)
        return model

    @timecall
    def train_model(self):
        model_choice = self.model_choice
        logger.info("Prepare data to train model")
        data = self.train_data
        target = data[self.label]
        target_log = np.log(target)
        train_set = data.drop(
            ['id', self.label], axis=1).astype(float)
        # categorical_features_indices = self.convert_to_categrorical_features(
        #     train_set)
        logger.info("Training model ....")
        features = train_set.columns.values.tolist()
        logger.debug("Features:" + str(len(features)))
        logger.debug(features)
        cat_features = ['vendor_id', 'store_and_fwd_flag', 'pickup_month',
                        'pickup_weekday', 'pickup_day', 'pickup_hour',
                        'pickup_whour', 'pickup_minute'
                        ]
        logger.debug("Categorial features:")
        logger.debug(cat_features)
        # logger.debug(train_set[cat_features].describe())
        X_train, X_test, Y_train, Y_test = train_test_split(
            train_set, target_log, train_size=0.85, random_state=1234)
        early_stopping_rounds = 50
        start = time.time()
        if model_choice == VW:
            self.model = self.vowpalwabbit_model()
            self.model.fit(X_train, Y_train)
        elif model_choice == CATBOOST:
            self.model = self.catboost_model()

            self.model.fit(
                X_train, Y_train, eval_set=(X_test, Y_test),
                cat_features=categorical_features_indices, verbose=True
            )
        elif model_choice == LIGHTGBM:
            self.model = self.lgbm_model()
            self.model.fit(
                X_train, Y_train, eval_set=[(X_test, Y_test)],
                eval_metric="rmse",
                early_stopping_rounds=early_stopping_rounds,
                verbose=10,
                # feature_name=features,
                # categorical_feature=cat_features
                # categorical_feature=categorical_features_indices
            )
        else:
            self.model = self.xgb_model()
            self.model.fit(
                X_train, Y_train, eval_set=[(X_test, Y_test)],
                eval_metric="rmse",
                early_stopping_rounds=early_stopping_rounds,
                verbose=early_stopping_rounds
            )

        end = time.time() - start
        logger.debug("Done training:" + str(end))
        if model_choice == LIGHTGBM:
            logger.debug("Predicting for:" + str(model_choice) +
                         ". Best round:" + str(self.model.best_iteration))
            y_pred = self.model.predict(
                X_test, num_iteration=self.model.best_iteration)
        elif model_choice == XGB:
            logger.debug("Predicting for:" + str(model_choice) +
                         ". Best round:" + str(self.model.best_iteration) +
                         ". N_tree_limit:" + str(self.model.best_ntree_limit))
            y_pred = self.model.predict(
                X_test, ntree_limit=self.model.best_ntree_limit)
        else:
            y_pred = self.model.predict(X_test)
        score = self.rmsle(Y_test.values, y_pred, log=True)
        score1 = self.rmsle(Y_test.values, y_pred, log=False)
        logger.debug("RMSLE score:" + str(score) +
                     " RMSLE without-log:" + str(score1))

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
                           verbose=early_stopping_rounds)
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
                      verbose=early_stopping_rounds)
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
            verbose=early_stopping_rounds
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
        data = self.eval_data.drop('id', axis=1).astype(float)
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

    # Build models for level 1
    def build_models_level1(self):
        logger.info('Bulding models level 1..')
        models = []
        # model_name = model.__class__.__name__
        # models.append(self.lgbm_model(random_state=123))
        # models.append(self.lgbm_model(random_state=789))
        models.append(self.xgb_model(learning_rate=0.03, random_state=456))
        # models.append(self.xgb_model(
        #     learning_rate=LEARNING_RATE, random_state=1000))
        models.append(self.lgbm_model(random_state=1024))
        return models

    # Kfold, train for each model, stack result
    @timecall
    def train_base_models(self):
        # Credit
        # to:https://dnc1994.com/2016/05/rank-10-percent-in-first-kaggle-competition-en/
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
        models = self.build_models_level1()
        logger.info("Training for model level 1 ...")
        # n_folds = len(models)
        n_folds = N_FOLDS
        kfolds = KFold(n_splits=n_folds, shuffle=True, random_state=321)
        S_train = np.zeros((X_in.shape[0], len(models)))
        S_test = np.zeros((T_in.shape[0], len(models)))
        logger.debug("X shape:" + str(X_in.shape) + " Y shape:" +
                     str(Y_in.shape) + " Test shape:" + str(T_in.shape))
        logger.debug("S_train shape:" + str(S_train.shape) + " S_test shape:" +
                     str(S_test.shape))
        all_rmse = 0
        early_stopping_rounds = 50
        start = time.time()
        for i in range(len(models)):
            model_name = models[i].__class__.__name__
            logger.debug("Base model " + str(i + 1) + ":" + model_name)
            S_test_i = np.zeros((T_in.shape[0], n_folds))
            model_rmse = 0
            for j, (train_idx, test_idx) in enumerate(kfolds.split(X_in)):
                model = copy.copy(models[i])
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
                        verbose=10
                    )
                    logger.debug("fold:" + str(j + 1) +
                                 " done training. Best round:" +
                                 str(model.best_ntree_limit) + " .Predicting for RMSLE")
                    y_pred = model.predict(
                        X_holdout, ntree_limit=model.best_ntree_limit)[:]
                    S_train[test_idx, i] = y_pred
                    S_test_i[:, j] = model.predict(
                        T_in, ntree_limit=model.best_ntree_limit)[:]
                elif model_name == 'LGBMRegressor':
                    model.fit(
                        X_train, Y_train, eval_set=[(X_holdout, y_holdout)],
                        eval_metric="rmse",
                        early_stopping_rounds=early_stopping_rounds,
                        verbose=10,
                        # feature_name=features,
                        # categorical_feature=cat_features
                        # categorical_feature=categorical_features_indices
                    )
                    logger.debug("fold:" + str(j + 1) +
                                 " done training. Best round:" +
                                 str(model.best_iteration) + " .Predicting for RMSLE")
                    y_pred = model.predict(
                        X_holdout, num_iteration=model.best_iteration)[:]
                    S_train[test_idx, i] = y_pred
                    S_test_i[:, j] = model.predict(
                        T_in, num_iteration=model.best_iteration)[:]
                else:
                    model.fit(X_train, y_train)
                    logger.debug("fold:" + str(j + 1) +
                                 " done training. Predicting for RMSLE")
                    y_pred = model.predict(X_holdout)[:]
                    S_train[test_idx, i] = y_pred
                    S_test_i[:, j] = model.predict(T_in)[:]

                rmse1 = self.rmsle(y_holdout, y_pred, log=False)
                model_rmse = model_rmse + rmse1
                all_rmse = all_rmse + rmse1
                logger.debug("fold:" + str(j + 1) + " rmse:" + str(rmse1))
                # cleanup memory
                del model
                # end of for j

            S_test[:, i] = S_test_i.mean(1)
            logger.debug("Model rmse:" + str(model_rmse / (j + 1)))
            # cleanup memory
            del S_test_i
            # end of for i

        end = time.time() - start
        logger.debug("All AVG rmse:" + str(all_rmse / (j + 1) / len(models)))
        logger.info("Done training base models:" + str(end))
        # print("Detect zero value")
        # print(np.where(S_train == 0))
        # print(np.where(S_test == 0))
        # save pre-train data
        self.save_pretrained_data(S_train, Y_in, S_test)
        return S_train, S_test

    # save pretrained data from train_base_models
    @timecall
    def save_pretrained_data(self, X_in, Y_in, T_in):
        logger.info("Saving pretrained data .. ")
        d_train = pd.DataFrame(data=X_in)
        d_train['label'] = Y_in
        d_train['id'] = self.train_data['id']
        d_train.to_csv(DATA_DIR + '/train_stack.csv', index=False)
        d_test = pd.DataFrame(data=T_in)
        d_test['id'] = self.eval_data['id']
        d_test.to_csv(DATA_DIR + '/test_stack.csv', index=False)

    # load pretrained data
    @timecall
    def load_pretrained_data(self):
        logger.info("Loading pretrained data .. ")
        d_train = pd.read_csv(DATA_DIR + '/train_stack.csv')
        d_test = pd.read_csv(DATA_DIR + '/test_stack.csv')
        return d_train, d_test

    @timecall
    def train_stack_model(self):
        logger.info("Prepare data to train stack model")
        S_train, S_test = self.load_pretrained_data()
        target_log = S_train['label']
        train_set = S_train.drop(['label', 'id'], axis=1)
        X_train, X_test, Y_train, Y_test = train_test_split(
            train_set, target_log, train_size=0.85, random_state=1234)
        model = LGBMRegressor(objective='regression_l2',
                              metric='l2_root',
                              n_estimators=N_ROUNDS,
                              #   n_estimators=10,
                              learning_rate=0.01,
                              num_leaves=1024,
                              #  max_bin=1024,
                              #  min_data_in_leaf=100,
                              nthread=-1, silent=False)
        early_stopping_rounds = 10
        start = time.time()
        model.fit(
            X_train, Y_train, eval_set=[(X_test, Y_test)],
            eval_metric="rmse",
            early_stopping_rounds=early_stopping_rounds,
            verbose=10,
            # feature_name=features,
                        # categorical_feature=cat_features
                        # categorical_feature=categorical_features_indices
        )

        end = time.time() - start
        logger.info("Done training for stack model:" + str(end))

        eval_set = S_test.drop(['id'], axis=1)
        logger.debug("Predicting for stack model" +
                     ". Best round:" + str(model.best_iteration))
        Y_eval_log = model.predict(
            eval_set, num_iteration=model.best_iteration)
        self.save_stacked_data(Y_eval_log)

    # Cross validation for stack model
    @timecall
    def stack_cv(self):
        logger.info("Prepare data to CV Stack model")
        S_train, S_test = self.load_pretrained_data()
        target_log = S_train['label']
        train_set = S_train.drop(['label', 'id'], axis=1)
        lgb_train = lgb.Dataset(train_set, target_log)
        params = {
            'objective': 'regression_l2',
            'metric': 'l2_root',
            'learning_rate': 0.01,
            'num_leaves': 1024,
            # 'max_bin': 1024,
            # 'min_data_in_leaf': 100,
            # 'nthread': -1,
            'verbose': 1
        }
        early_stopping_rounds = 10
        logger.debug("Cross validating for stack model ...")
        cv_results = lgb.cv(params, lgb_train, num_boost_round=200, nfold=5,
                            metrics="rmse", shuffle=True,
                            early_stopping_rounds=early_stopping_rounds,
                            verbose_eval=10, show_stdv=True, seed=1000)
        logger.debug("round:" + str(len(cv_results['rmse-mean'])) + " .rmse:" +
                     str(cv_results['rmse-mean'][-1]) + "+" + str(cv_results['rmse-stdv'][-1]))

    @timecall
    def save_stacked_data(self, Y_eval_log):
        logger.info("Saving submission for stack model to disk")
        Y_eval = np.exp(Y_eval_log.ravel())
        eval_output = self.eval_data.copy()
        eval_output.loc[:, self.label] = Y_eval
        today = str(dtime.date.today())
        eval_output[['id', self.label]].to_csv(
            DATA_DIR + '/' + today + '-submission.csv.gz', index=False,
            compression='gzip')


# ---------------- Main -------------------------
if __name__ == "__main__":
    option = 6
    model_choice = LIGHTGBM
    logger = logging.getLogger('newyork-taxi-duration')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(LOG_LEVEL)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(DATA_DIR + '/model.log', mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    base_class = TaxiTripDuration(LABEL, model_choice)
    # Load and preprocessed data
    if option == 1:
        base_class.load_data()
        base_class.check_null_data()
        quit()
        base_class.preprocess_data()
        base_class.check_null_data()
        base_class.feature_correlation()
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

    # Load process data and train model with stacking model:
    elif option == 5:
        base_class.load_preprocessed_data()
        base_class.train_base_models()
        base_class.train_stack_model()

    # load pretrain-data from train_base_model and train with stacking model
    elif option == 6:
        base_class.load_preprocessed_data()
        base_class.train_stack_model()

    # Load process data and search for best model parameters
    elif option == 10:
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

    # ------------- Stack model
    elif option == 53:
        base_class.stack_cv()

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
        base_class.importance_features()
        # base_class.plot_ft_importance()

    logger.info("Done!")
