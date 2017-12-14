# https://www.kaggle.com/the1owl/surprise-me
"""
Contributions from:
DSEverything - Mean Mix - Math, Geo, Harmonic (LB 0.493) 
https://www.kaggle.com/dongxu027/mean-mix-math-geo-harmonic-lb-0-493
JdPaletto - Surprised Yet? - Part2 - (LB: 0.503)
https://www.kaggle.com/jdpaletto/surprised-yet-part2-lb-0-503
hklee - weighted mean comparisons, LB 0.497, 1ST
https://www.kaggle.com/zeemeen/weighted-mean-comparisons-lb-0-497-1st

Also all comments for changes, encouragement, and forked scripts rock

Keep the Surprise Going
"""

import glob
import re
import numpy as np
import pandas as pd
from sklearn import *
from datetime import datetime
DATA_DIR = 'data-temp'
data = {
    'tra': pd.read_csv(DATA_DIR + '/air_visit_data.csv'),
    'as': pd.read_csv(DATA_DIR + '/air_store_info.csv'),
    'hs': pd.read_csv(DATA_DIR + '/hpg_store_info.csv'),
    'ar': pd.read_csv(DATA_DIR + '/air_reserve.csv'),
    'hr': pd.read_csv(DATA_DIR + '/hpg_reserve.csv'),
    'id': pd.read_csv(DATA_DIR + '/store_id_relation.csv'),
    'tes': pd.read_csv(DATA_DIR + '/sample_submission.csv'),
    'hol': pd.read_csv(DATA_DIR + '/date_info.csv').rename(columns={'calendar_date': 'visit_date'})
}

data['hr'] = pd.merge(data['hr'], data['id'], how='inner', on=['hpg_store_id'])

for df in ['ar', 'hr']:
    data[df]['visit_datetime'] = pd.to_datetime(data[df]['visit_datetime'])
    data[df]['visit_datetime'] = data[df]['visit_datetime'].dt.date
    data[df]['reserve_datetime'] = pd.to_datetime(data[df]['reserve_datetime'])
    data[df]['reserve_datetime'] = data[df]['reserve_datetime'].dt.date
    data[df]['reserve_datetime_diff'] = data[df].apply(lambda r: (
        r['visit_datetime'] - r['reserve_datetime']).days, axis=1)
    tmp1 = data[df].groupby(['air_store_id', 'visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].sum(
    ).rename(columns={'visit_datetime': 'visit_date', 'reserve_datetime_diff': 'rs1', 'reserve_visitors': 'rv1'})
    tmp2 = data[df].groupby(['air_store_id', 'visit_datetime'], as_index=False)[['reserve_datetime_diff', 'reserve_visitors']].mean(
    ).rename(columns={'visit_datetime': 'visit_date', 'reserve_datetime_diff': 'rs2', 'reserve_visitors': 'rv2'})
    data[df] = pd.merge(tmp1, tmp2, how='inner', on=[
                        'air_store_id', 'visit_date'])

data['tra']['visit_date'] = pd.to_datetime(data['tra']['visit_date'])
data['tra']['dow'] = data['tra']['visit_date'].dt.dayofweek
data['tra']['year'] = data['tra']['visit_date'].dt.year
data['tra']['month'] = data['tra']['visit_date'].dt.month
data['tra']['visit_date'] = data['tra']['visit_date'].dt.date

data['tes']['visit_date'] = data['tes']['id'].map(
    lambda x: str(x).split('_')[2])
data['tes']['air_store_id'] = data['tes']['id'].map(
    lambda x: '_'.join(x.split('_')[:2]))
data['tes']['visit_date'] = pd.to_datetime(data['tes']['visit_date'])
data['tes']['dow'] = data['tes']['visit_date'].dt.dayofweek
data['tes']['year'] = data['tes']['visit_date'].dt.year
data['tes']['month'] = data['tes']['visit_date'].dt.month
data['tes']['visit_date'] = data['tes']['visit_date'].dt.date

unique_stores = data['tes']['air_store_id'].unique()
stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [
                   i] * len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)

# sure it can be compressed...
# tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)[
#     'visitors'].min().rename(columns={'visitors': 'min_visitors'})
# stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
# tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)[
#     'visitors'].mean().rename(columns={'visitors': 'mean_visitors'})
# stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
# tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)[
#     'visitors'].median().rename(columns={'visitors': 'median_visitors'})
# stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
# tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)[
#     'visitors'].max().rename(columns={'visitors': 'max_visitors'})
# stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])
# tmp = data['tra'].groupby(['air_store_id', 'dow'], as_index=False)[
#     'visitors'].count().rename(columns={'visitors': 'count_observations'})
# stores = pd.merge(stores, tmp, how='left', on=['air_store_id', 'dow'])

stores = pd.merge(stores, data['as'], how='left', on=['air_store_id'])
lbl = preprocessing.LabelEncoder()
stores['air_genre_name'] = lbl.fit_transform(stores['air_genre_name'])
stores['air_area_name'] = lbl.fit_transform(stores['air_area_name'])

data['hol']['visit_date'] = pd.to_datetime(data['hol']['visit_date'])
data['hol']['day_of_week'] = lbl.fit_transform(data['hol']['day_of_week'])
data['hol']['visit_date'] = data['hol']['visit_date'].dt.date
train = pd.merge(data['tra'], data['hol'], how='left', on=['visit_date'])
test = pd.merge(data['tes'], data['hol'], how='left', on=['visit_date'])

train = pd.merge(train, stores, how='left', on=['air_store_id', 'dow'])
test = pd.merge(test, stores, how='left', on=['air_store_id', 'dow'])

for df in ['ar', 'hr']:
    train = pd.merge(train, data[df], how='left', on=[
                     'air_store_id', 'visit_date'])
    test = pd.merge(test, data[df], how='left', on=[
                    'air_store_id', 'visit_date'])

train['id'] = train.apply(lambda r: '_'.join(
    [str(r['air_store_id']), str(r['visit_date'])]), axis=1)

# train['total_reserv_sum'] = train['rv1_x'] + train['rv1_y']
# train['total_reserv_mean'] = (train['rv2_x'] + train['rv2_y']) / 2
# train['total_reserv_dt_diff_mean'] = (train['rs2_x'] + train['rs2_y']) / 2

# test['total_reserv_sum'] = test['rv1_x'] + test['rv1_y']
# test['total_reserv_mean'] = (test['rv2_x'] + test['rv2_y']) / 2
# test['total_reserv_dt_diff_mean'] = (test['rs2_x'] + test['rs2_y']) / 2

col = [c for c in train if c not in [
    'id', 'air_store_id', 'visit_date', 'visitors']]
train = train.fillna(-1)
test = test.fillna(-1)

train.to_csv(DATA_DIR + '/train_2.csv', index=False)
test.to_csv(DATA_DIR + '/test_2.csv', index=False)
print("Done!!!")