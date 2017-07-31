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
# # XGB
from xgboost.sklearn import XGBRegressor
import xgboost as xgb
from xgboost import plot_importance
# # CatBoost
# from catboost import Pool, CatBoostRegressor, cv, CatboostIpythonWidget
# System
import datetime as dtime
from datetime import datetime
import sys
from inspect import getsourcefile
import os.path
import re
import time
from profilehooks import timecall
# Other
# from geographiclib.geodesic import Geodesic
# import osmnx as ox
# import networkx as nx

pd.options.display.float_format = '{:,.4f}'.format
# Input data files are available in the DATA_DIR directory.
DATA_DIR = "data-temp"
LABEL = 'trip_duration'
N_FOLDS = 5
# Learning param
# 'learning_rate': 0.1, 'min_child_weight': 5, 'max_depth': 10 => Best
# 'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 10
# 'max_depth': 5, 'learning_rate': 0.1, 'min_child_weight': 5
LEARNING_RATE = 0.1
MIN_CHILD_WEIGHT = 5
MAX_DEPTH = 10
N_ROUNDS = 10000
# N_ROUNDS = 10


class TaxiTripDuration():
    def __init__(self, label):
        self.label = label

    @timecall
    def load_data(self):
        print("Loading data ....")
        start = time.time()
        label = self.label
        # Load data. Download
        # from:https://www.kaggle.com/c/nyc-taxi-trip-duration/data
        train_data = pd.read_csv(DATA_DIR + "/train.csv")
        eval_data = pd.read_csv(DATA_DIR + "/test.csv")
        # Download from:
        # https://www.kaggle.com/oscarleo/new-york-city-taxi-with-osrm
        train_osm = pd.read_csv(DATA_DIR + "/fastest_route_train.csv")
        eval_osm = pd.read_csv(DATA_DIR + "/fastest_route_test.csv")
        print("train size:", train_data.shape, " test size:", eval_data.shape)
        print("train_osm size:", train_osm.shape,
              " test osm size:", eval_osm.shape)

        print("Merging  2 data sets ...")
        col_use = ['id', 'total_distance', 'total_travel_time',
                   'number_of_steps',
                   'starting_street', 'end_street']
        train_osm_data = train_osm[col_use]
        eval_osm_data = eval_osm[col_use]
        train_data = train_osm_data.join(train_data.set_index('id'), on='id')
        # Cleanup data
        train_data = train_data[train_data[self.label] < 1800000]
        eval_data = eval_osm_data.join(eval_data.set_index('id'), on='id')
        features = eval_data.columns.values
        self.target = train_data[label]
        self.combine_data = pd.concat(
            [train_data[features], eval_data], keys=['train', 'eval'])
        print("combine data:", len(self.combine_data))
        end = time.time() - start
        print("Data loaded")

    @timecall
    def load_preprocessed_data(self):
        print("Loading preprocessed data ....")
        label = self.label
        # Load data. Download
        # from:https://www.kaggle.com/c/nyc-taxi-trip-duration/data
        self.train_data = pd.read_csv(DATA_DIR + "/train_pre.csv")
        self.eval_data = pd.read_csv(DATA_DIR + "/test_pre.csv")
        print("train size:", self.train_data.shape,
              " test size:", self.eval_data.shape)
        # features = eval_data.columns.values
        # self.target = train_data[label]
        print("Data loaded")

    def check_null_data(self, data=None):
        print("Check for null data")
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
        # print(missing_data)
        print(high_percent_miss_data)
        miss_data_cols = high_percent_miss_data.index.values
        return miss_data_cols

    def fillnan(self):
        print("FillNaN ...")
        data = self.combine_data
        data.loc[:, 'total_distance'].fillna(1, inplace=True)
        data.loc[:, 'number_of_steps'].fillna(1, inplace=True)

    def cleanup_data(self):
        self.train_data = self.train_data[self.train_data[self.label] < 1800000]

    def convert_datetime(self):
        print("Convert datetime ...")
        data = self.combine_data
        data.loc[:, 'datetime_obj'] = pd.to_datetime(data['pickup_datetime'])
        data.loc[:, 'pickup_year'] = data['datetime_obj'].dt.year
        data.loc[:, 'pickup_month'] = data['datetime_obj'].dt.month
        data.loc[:, 'pickup_weekday'] = data['datetime_obj'].dt.weekday
        data.loc[:, 'pickup_day'] = data['datetime_obj'].dt.day
        data.loc[:, 'pickup_hour'] = data['datetime_obj'].dt.hour
        data.loc[:, 'pickup_minute'] = data['datetime_obj'].dt.minute

    def convert_store_and_fwd_flag(self):
        print("Convert store_and_fwd_flag ...")
        data = self.combine_data
        col = 'store_and_fwd_flag'
        data_dict = {'Y': 1, 'N': 0}
        data_tf = data[col].map(data_dict)
        data.loc[:, col].update(data_tf)

    def convert_starting_street(self):
        print("Convert starting_street ...")
        data = self.combine_data
        col = 'starting_street'
        data_not_null = data[data[col].notnull()]
        le = LabelEncoder()
        data_tf = le.fit_transform(data_not_null[col])
        col_tf = col + '_tf'
        # data_not_null.loc[:, col_tf] = data_tf
        data.loc[data[col].notnull(), col_tf] = data_tf

    def convert_end_street(self):
        print("Convert end_street ...")
        data = self.combine_data
        col = 'end_street'
        data_not_null = data[data[col].notnull()]
        le = LabelEncoder()
        data_tf = le.fit_transform(data_not_null[col])
        col_tf = col + '_tf'
        # data_not_null.loc[:, col_tf] = data_tf
        data.loc[data[col].notnull(), col_tf] = data_tf

    def drop_unused_cols(self):
        data = self.combine_data
        data.drop('pickup_datetime', axis=1, inplace=True)
        data.drop('datetime_obj', axis=1, inplace=True)
        data.drop('starting_street', axis=1, inplace=True)
        data.drop('end_street', axis=1, inplace=True)

    @timecall
    def feature_starting_street(self):
        data = self.combine_data
        col = 'starting_street_tf'
        print("Feature enginering ", col)
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
        print("Feature enginering ", col)
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

    # credit:
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

    @timecall
    def feature_haversin(self):
        print("Feature engineering: haversine_distance")
        data = self.combine_data
        data.loc[:, 'haversine_distance'] = self.haversine_np(
            data['pickup_longitude'], data['pickup_latitude'],
            data['dropoff_longitude'], data['dropoff_latitude'])

    def estimate_total_distance(self):
        print("Estimating total_distance ... ")
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
        print("Feature engineering: total_distance")
        data = self.combine_data
        col1 = 'haversine_distance'
        col = 'total_distance'
        print("check for haversine_distance=0")
        data.loc[data[col1] == 0, col] = 0
        data.loc[data[col1] == 0, 'number_of_streets'] = 1
        print("check for total_distance=0 and haversine_distance=0")
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
        # data_speed.loc[data_speed[self.label] == 0, col] = 0
        # data_speed_not_zero = data_speed.loc[data_speed[self.label] > 0]
        # data_speed_not_zero.loc[:, col] = data_speed_not_zero[distance_col] / \
        #     data_speed_not_zero[self.label]
        # data_speed.update(data_speed_not_zero)
        data_speed.loc[:, col] = data_speed[distance_col] / \
            data_speed[self.label]
        data_speed.loc[:, col].fillna(data_speed[col].mean(), inplace=True)
        return data_speed

    def speed_mean_by_col(self, col, suffix, data_speed):
        print("Speed mean by ", col, suffix)
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
        print("Calculating speed_mean by total_distance for each feature")
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

        # print("Calculating speed_mean by haversine for each feature")
        # distance_col = 'haversine_distance'
        # suffix = '_hv_speed_mean'
        # data_speed = self.cal_speed(distance_col)
        # col = 'pickup_hour'
        # self.speed_mean_by_col(col, suffix, data_speed)
        # col = 'pickup_weekday'
        # self.speed_mean_by_col(col, suffix, data_speed)
        # col = 'pickup_day'
        # self.speed_mean_by_col(col, suffix, data_speed)
        # col = 'pickup_month'
        # self.speed_mean_by_col(col, suffix, data_speed)

    @timecall
    def duration_mean_by_col(self, col):
        print("Duration mean by ", col)
        col_speed_mean = col + '_speed_mean'
        col_duration_mean = col + '_duration_mean'
        data = self.combine_data
        data.loc[:, col_duration_mean] = data['total_distance'] / \
            data[col_speed_mean]
        data.loc[:, col_duration_mean].fillna(data[col_duration_mean].mean(), inplace=True)

    @timecall
    def haversine_duration_mean_by_col(self, col):
        print("Duration mean by ", col)
        col_speed_mean = col + '_hv_speed_mean'
        col_duration_mean = col + '_hv_duration_mean'
        data = self.combine_data
        data.loc[:, col_duration_mean] = data['haversine_distance'] / \
            data[col_speed_mean]

    @timecall
    def feature_duration_mean(self):
        print("Calculating duration_mean by total_distance for each feature")
        self.duration_mean_by_col('pickup_hour')
        self.duration_mean_by_col('pickup_weekday')
        self.duration_mean_by_col('pickup_day')
        self.duration_mean_by_col('pickup_month')

        self.duration_mean_by_col('starting_street_tf')
        self.duration_mean_by_col('end_street_tf')

        # print("Calculating duration_mean by haversine for each feature")
        # self.haversine_duration_mean_by_col('pickup_hour')
        # self.haversine_duration_mean_by_col('pickup_weekday')
        # self.haversine_duration_mean_by_col('pickup_day')
        # self.haversine_duration_mean_by_col('pickup_month')

    @timecall
    def feature_distance_by_step(self):
        print("Calculating total_distance/number_of_steps ...")
        data = self.combine_data
        col = ' distance_per_step'
        data.loc[:, col] = data['total_distance'] / data['number_of_steps']
        data.loc[:, col].fillna(data[col].mean(), inplace=True)

    @timecall
    def feature_haversine_distance_by_step(self):
        print("Calculating haversine_distance/number_of_steps ...")
        data = self.combine_data
        col = ' hv_distance_per_step'
        data.loc[:, col] = data['haversine_distance'] / \
            data['number_of_steps']
        data.loc[:, col].fillna(data[col].mean(), inplace=True)    

    @timecall
    def preprocess_data(self):
        print("Preproccesing data ...")
        self.fillnan()
        self.convert_datetime()
        self.convert_starting_street()
        self.convert_end_street()
        self.convert_store_and_fwd_flag()
        self.feature_haversin()
        # There is no NaN starting_street and end_street => no need to feature enginering
        # self.feature_starting_street()
        # self.feature_end_street()
        self.feature_speed_mean()
        # self.feature_duration_mean()
        self.feature_distance_by_step()
        # self.feature_haversine_distance_by_step()

        # Drop unsed columns
        self.drop_unused_cols()
        print(self.combine_data.columns.values)

        # Save preprocess data
        train_set = self.combine_data.loc['train'].copy()
        train_set.loc[:, self.label] = self.target
        eval_set = self.combine_data.loc['eval']
        print("Saving train set ...")
        train_set.to_csv(DATA_DIR + '/train_pre.csv', index=False)
        print("Saving eval set ...")
        eval_set.to_csv(DATA_DIR + '/test_pre.csv', index=False)
        # Save to class variable for later use
        self.train_data = train_set
        self.eval_data = eval_set
        #Freeup memory
        del self.combine_data

    def rmsle(self, y, y_pred, log=True):
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
        print("Prepare data for  model")
        data = self.train_data
        target = data[self.label]
        target_log = np.log(target)
        train_set = data.drop(
            ['id', 'pickup_year', self.label], axis=1).astype(float)
        X_train, X_test, Y_train, Y_test = train_test_split(
            train_set, target_log, test_size=10000, random_state=1234)
        param_grid = {"max_depth": [1, 5, 10],
                      'learning_rate': [0.1, 0.3],
                      'min_child_weight':  [1, 5, 10]
                      }
        model = XGBRegressor(n_estimators=500, max_depth=5,
                             learning_rate=0.1, min_child_weight=1, n_jobs=-1)
        print("Searching for best params")
        grid_search = GridSearchCV(model, param_grid, n_jobs=1, cv=5)
        grid_search.fit(X_test, Y_test)
        print(grid_search.best_params_)
        # Best model:'learning_rate': 0.1, 'min_child_weight': 5, 'max_depth':
        # 10

    @timecall
    def train_model(self):
        print("Prepare data to train model")
        data = self.train_data
        target = data[self.label]
        target_log = np.log(target)
        train_set = data.drop(
            ['id', 'pickup_year', self.label], axis=1).astype(float)
        X_train, X_test, Y_train, Y_test = train_test_split(
            train_set, target_log, train_size=0.85, random_state=1234)
        self.model = XGBRegressor(n_estimators=N_ROUNDS, max_depth=MAX_DEPTH,
                                  learning_rate=LEARNING_RATE, min_child_weight=MIN_CHILD_WEIGHT, n_jobs=-1)
        print("Training model ....")
        print(train_set.columns.values)
        start = time.time()
        early_stopping_rounds = 50
        self.model.fit(
            X_train, Y_train, eval_set=[(X_train, Y_train), (X_test, Y_test)],
            eval_metric="rmse", early_stopping_rounds=early_stopping_rounds,
            verbose=early_stopping_rounds
        )
        end = time.time() - start
        print("Done training:", end)
        y_pred = self.model.predict(X_test)
        score = self.rmsle(Y_test.values, y_pred)
        score1 = self.rmsle(Y_test.values, y_pred, log=False)
        print("RMSLE score:", score, " RMSLE without-log:", score1)

    # Train using Kflow
    @timecall
    def train_kfold_single(self):
        print("Prepare data to train model")
        data = self.train_data
        target = data[self.label]
        target_log = np.log(target)
        train_set = data.drop(
            ['id', 'pickup_year', self.label], axis=1).astype(float)
        total_rmse = 0
        kfolds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=321)
        self.model = XGBRegressor(n_estimators=N_ROUNDS, max_depth=MAX_DEPTH,
                                  learning_rate=LEARNING_RATE, min_child_weight=MIN_CHILD_WEIGHT, n_jobs=-1)
        X = train_set
        Y = target_log.values
        early_stopping_rounds = 50
        total_time = 0
        for j, (train_idx, test_idx) in enumerate(kfolds.split(X)):
            print("Round:", j + 1)
            start = time.time()
            X_train = X.iloc[train_idx]
            Y_train = Y[train_idx]
            X_test = X.iloc[test_idx]
            Y_test = Y[test_idx]
            self.model.fit(X_train, Y_train, eval_set=[(X_train, Y_train), (X_test, Y_test)],
                           eval_metric="rmse", early_stopping_rounds=early_stopping_rounds,
                           verbose=early_stopping_rounds)
            end = time.time() - start
            total_time = total_time + end
            print("Done training for round:", j +
                  1, " time:", end, "/", total_time)
            y_pred = self.model.predict(X_test)
            rmse1 = self.rmsle(Y_test, y_pred, log=False)
            total_rmse = total_rmse + rmse1
            print("rmsle:", rmse1)
        print("Avg rmse:", total_rmse / (j + 1))
        print("Total training time:", total_time)

    # Train using Kflow and arrgregate trained models
    @timecall
    def train_kfold_aggregate(self):
        print("Prepare data to train model")
        data = self.train_data
        target = data[self.label]
        target_log = np.log(target)
        train_set = data.drop(
            ['id', 'pickup_year', self.label], axis=1).astype(float)
        total_rmse = 0
        kfolds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=321)
        model = XGBRegressor(n_estimators=N_ROUNDS, max_depth=MAX_DEPTH,
                             learning_rate=LEARNING_RATE, min_child_weight=MIN_CHILD_WEIGHT, n_jobs=-1)
        X = train_set
        Y = target_log.values
        T = self.eval_data.drop(
            ['id', 'pickup_year'], axis=1).astype(float)
        S_train = np.zeros((X.shape[0], N_FOLDS))
        S_test = np.zeros((T.shape[0], N_FOLDS))
        early_stopping_rounds = 50
        total_time = 0
        for j, (train_idx, test_idx) in enumerate(kfolds.split(X)):
            print("Round:", j + 1)
            start = time.time()
            X_train = X.iloc[train_idx]
            Y_train = Y[train_idx]
            X_test = X.iloc[test_idx]
            Y_test = Y[test_idx]
            model.fit(X_train, Y_train, eval_set=[(X_train, Y_train), (X_test, Y_test)],
                      eval_metric="rmse", early_stopping_rounds=early_stopping_rounds,
                      verbose=early_stopping_rounds)
            end = time.time() - start
            total_time = total_time + end
            print("Done training for round:", j +
                  1, " time:", end, "/", total_time)
            y_pred = model.predict(X_test)
            rmse1 = self.rmsle(Y_test, y_pred, log=False)
            total_rmse = total_rmse + rmse1
            print("rmsle:", rmse1)
            print("Saving Y_pred for round:", j + 1)
            S_train[:, j] = model.predict(X)
            print("Saving Y_eval for round:", j + 1)
            S_test[:, j] = model.predict(T)
        print("Avg rmse:", total_rmse / (j + 1))
        print("Total training time:", total_time)
        print(S_train[:5])
        return S_train, S_test

    # Stack train from kfold
    @timecall
    def train_predict_kfold_aggregate(self):
        print("Training model using kfold ang aggregate results")
        train_set, test_set = self.train_kfold_aggregate()
        target = self.train_data[self.label]
        target_log = np.log(target)

        X_train, X_test, Y_train, Y_test = train_test_split(
            train_set, target_log, train_size=0.85, random_state=1234)
        self.model = XGBRegressor(n_estimators=N_ROUNDS, max_depth=MAX_DEPTH,
                                  learning_rate=LEARNING_RATE, min_child_weight=MIN_CHILD_WEIGHT, n_jobs=-1)
        print("Training stack model ....")
        start = time.time()
        early_stopping_rounds = 50
        self.model.fit(
            X_train, Y_train, eval_set=[(X_train, Y_train), (X_test, Y_test)],
            eval_metric="rmse", early_stopping_rounds=early_stopping_rounds,
            verbose=early_stopping_rounds
        )
        end = time.time() - start
        print("Done training:", end)
        y_pred = self.model.predict(X_test)
        score = self.rmsle(Y_test.values, y_pred)
        score1 = self.rmsle(Y_test.values, y_pred, log=False)
        print("RMSLE score:", score, " RMSLE without-log:", score1)
        print("Predict for eval set ..")
        Y_eval_log = self.model.predict(test_set)
        Y_eval = np.exp(Y_eval_log.ravel())
        print("Saving prediction to disk")
        eval_output = pd.DataFrame(
            {'id': self.eval_data['id'], self.label: Y_eval}, columns=['id', self.label])
        print("Eval data:", len(eval_output))
        today = str(dtime.date.today())
        eval_output.to_csv(
            DATA_DIR + '/' + today + '-submission.csv.gz', index=False, compression='gzip')

    @timecall
    def predict_save(self):
        print("Predicting for eval data ..")
        data = self.eval_data.drop(['id', 'pickup_year'], axis=1).astype(float)
        Y_eval_log = self.model.predict(data)
        Y_eval = np.exp(Y_eval_log.ravel())
        eval_output = self.eval_data.copy()
        eval_output.loc[:, self.label] = Y_eval
        print("Saving prediction to disk")
        today = str(dtime.date.today())
        eval_output[['id', self.label]].to_csv(
            DATA_DIR + '/' + today + '-submission.csv.gz', index=False, compression='gzip')

    def importance_features(self):
        print("Prepare data to train model")
        threshold = 0
        data = self.train_data
        target = data[self.label]
        target_log = np.log(target)
        train_set = data.drop(
            ['id', 'pickup_year', self.label], axis=1).astype(float)
        features_score = pd.Series(
            self.model.feature_importances_, index=train_set.columns.values)
        # print("Feature importance:", features_score.describe())
        print("Feature importance:")
        print(features_score.sort_values())
        # Drop features with score below threshold
        # features = features_score[features_score > threshold].index
        # features = features_score[features_score >=
        # np.percentile(features_score, threshold)].index
        # print("Remain features:", len(features))
        return

    def plot_ft_importance(self):
        # print(self.model.feature_importances_)
        self.importance_features()
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111)
        plot_importance(self.model, ax=ax)
        plt.savefig(DATA_DIR + '/feature_importance.png')
        graph = xgb.to_graphviz(self.model)
        graph.render()


# ---------------- Main -------------------------
if __name__ == "__main__":
    option = 0
    base_class = TaxiTripDuration(LABEL)
    # Load and preprocessed data
    if option == 1:
        base_class.load_data()
        base_class.check_null_data()
        base_class.preprocess_data()
        base_class.check_null_data()
        # base_class.cleanup_data()
        # Search for best model params based on current dataset
        # base_class.search_best_model_params()
    # Load process data and train model
    elif option == 2:
        base_class.load_preprocessed_data()
        base_class.cleanup_data()
        base_class.train_model()
        base_class.predict_save()
        base_class.plot_ft_importance()
    # Load process data and train model with Kfold
    elif option == 3:
        base_class.load_preprocessed_data()
        base_class.cleanup_data()
        base_class.train_kfold_single()
        base_class.predict_save()
        base_class.plot_ft_importance()
    # Load process data and train model with Kfold, aggregate result then
    # train again with aggregated data
    # Note: LB lower than Kfold single
    elif option == 4:
        base_class.load_preprocessed_data()
        base_class.cleanup_data()
        base_class.train_predict_kfold_aggregate()

    # Load process data and search for best model parameters
    elif option == 10:
        base_class.load_preprocessed_data()
        base_class.cleanup_data()
        base_class.search_best_model_params()

    # combine preprocess and training model
    else:
        base_class.load_data()
        base_class.check_null_data()
        base_class.preprocess_data()
        base_class.check_null_data()
        base_class.cleanup_data()
        base_class.train_model()
        base_class.predict_save()
        base_class.plot_ft_importance()
