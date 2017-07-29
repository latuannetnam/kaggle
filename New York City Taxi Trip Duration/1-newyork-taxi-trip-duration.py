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
        train_osm = pd.read_csv(DATA_DIR + "/train_2.csv")
        eval_osm = pd.read_csv(DATA_DIR + "/test_2.csv")
        print("train size:", train_data.shape, " test size:", eval_data.shape)
        print("train_osm size:", train_osm.shape,
              " test osm size:", eval_osm.shape)

        print("Merging  2 data sets ...")
        col_use = ['id', 'total_distance', 'number_of_streets',
                   'starting_street', 'end_street']
        train_osm_data = train_osm[col_use]
        eval_osm_data = eval_osm[col_use]
        train_data = train_data.join(train_osm_data.set_index('id'), on='id')
        eval_data = eval_data.join(eval_osm_data.set_index('id'), on='id')
        features = eval_data.columns.values
        self.target = train_data[label]
        self.combine_data = pd.concat(
            [train_data[features], eval_data], keys=['train', 'eval'])
        print("combine data:", len(self.combine_data))
        end = time.time() - start
        print("Data loaded")

    def check_null_data(self, data):
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
        data.loc[:, 'total_distance'].fillna(0, inplace=True)
        data.loc[:, 'number_of_streets'].fillna(1, inplace=True)

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

    @timecall
    def feature_haversin(self):
        print("Feature engineering: haversine_distance")
        data = self.combine_data
        haversine_distance = data.apply(lambda row:
                                        self.haversine(
                                            row['pickup_latitude'], row['pickup_longitude'],
                                            row['dropoff_latitude'], row['dropoff_longitude']), axis=1)
        data.loc[:, 'haversine_distance'] = haversine_distance

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

    def cal_speed_row(self, row):
        if row[self.label] == 0:
            return 0
        else:
            return row['total_distance'] / row[self.label]

    def cal_speed(self):
        col = 'speed'
        data_speed = self.combine_data.loc['train'].copy()
        data_speed.loc[:, self.label] = self.target
        data_speed_values = data_speed.apply(
            lambda row: self.cal_speed_row(row), axis=1)
        data_speed.loc[:, col] = data_speed_values
        return data_speed

    def speed_mean_by_col(self, col, data_speed):
        print("Speed mean by ", col)
        data = self.combine_data
        group_col = 'speed'
        data_grp = data_speed[[col, group_col]]
        data_st = data_grp.groupby(
            col, as_index=False).mean().sort_values(col).reset_index()
        data_sr = pd.Series(data_st[group_col], index=data_st[col])
        data_dict = data_sr.to_dict()
        col_mean = col + '_' + 'speed_mean'
        data.loc[:, col_mean] = data[col].map(data_dict)

    def speed_mean_by_hour(self, data_speed):
        col = 'pickup_hour'
        self.speed_mean_by_col(col, data_speed)

    def speed_mean_by_weekday(self, data_speed):
        col = 'pickup_weekday'
        self.speed_mean_by_col(col, data_speed)

    def speed_mean_by_day(self, data_speed):
        col = 'pickup_day'
        self.speed_mean_by_col(col, data_speed)

    def speed_mean_by_month(self, data_speed):
        col = 'pickup_month'
        self.speed_mean_by_col(col, data_speed)

    @timecall
    def cal_speed_mean(self):
        print("Calculating speed_mean for each feature")
        data_speed = self.cal_speed()
        self.speed_mean_by_hour(data_speed)
        self.speed_mean_by_weekday(data_speed)
        self.speed_mean_by_day(data_speed)
        self.speed_mean_by_month(data_speed)

    @timecall
    def duration_mean_by_col(self, col):
        print("Duration mean by ", col)
        col_speed_mean = col + '_speed_mean'
        col_duration_mean = col + '_duration_mean'
        data = self.combine_data
        duration_mean = data.apply(
            lambda row: row['total_distance'] / row[col_speed_mean], axis=1)
        data.loc[:, col_duration_mean] = duration_mean

    @timecall
    def cal_duration_mean(self):
        print("Calculating duration_mean for each feature")
        self.duration_mean_by_col('pickup_hour')
        self.duration_mean_by_col('pickup_weekday')
        self.duration_mean_by_col('pickup_day')
        self.duration_mean_by_col('pickup_month')

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

    @timecall
    def preprocess_data(self):
        print("Preproccesing data ...")
        self.fillnan()
        self.convert_datetime()
        self.convert_starting_street()
        self.convert_end_street()
        self.convert_store_and_fwd_flag()
        self.feature_haversin()
        self.feature_total_distance()
        self.feature_starting_street()
        self.feature_end_street()
        self.cal_speed_mean()
        self.cal_duration_mean()

        # Drop unsed columns
        self.drop_unused_cols()
        print(self.combine_data.columns.values)

        # Save preprocess data
        train_set = self.combine_data.loc['train'].copy()
        train_set.loc[:, self.label] = self.target
        eval_set = self.combine_data.loc['eval']
        train_set.to_csv(DATA_DIR + '/train_pre.csv', index=False)
        eval_set.to_csv(DATA_DIR + '/test_pre.csv', index=False)


# ---------------- Main -------------------------
if __name__ == "__main__":
    option = 1
    base_class = TaxiTripDuration(LABEL)
    if option == 1:
        base_class.load_data()
        base_class.preprocess_data()
