#
# This script is inspired by this discussion:
# https://www.kaggle.com/c/zillow-prize-1/discussion/33710
# credit:
# https://www.kaggle.com/danieleewww/xgboost-without-outliers-lb-0-06463/code

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
DATA_DIR = "data-temp"
print("Loading data ...")
properties = pd.read_csv(DATA_DIR + "/properties_2016.csv")
train = pd.read_csv(DATA_DIR + "/train_2016_v2.csv")
for c in properties.columns:
    properties[c] = properties[c].fillna(-1)
    if properties[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(properties[c].values))
        properties[c] = lbl.transform(list(properties[c].values))
print("Merging data ...")
train_df = train.merge(properties, how='left', on='parcelid')
x_train = train_df.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
x_test = properties.drop(['parcelid'], axis=1)
# shape
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))

# drop out ouliers
train_df = train_df[train_df.logerror > -0.4]
train_df = train_df[train_df.logerror < 0.42]
x_train = train_df.drop(['parcelid', 'logerror', 'transactiondate'], axis=1)
y_train = train_df["logerror"].values.astype(np.float32)
y_mean = np.mean(y_train)

print('After removing outliers:')
print('Shape train: {}\nShape test: {}'.format(x_train.shape, x_test.shape))


# xgboost params
xgb_params = {
    'eta': 0.033,
    'max_depth': 6,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'base_score': y_mean,
    'silent': 1
}

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)

print("Cross validation ...")
# cross-validation
cv_result = xgb.cv(xgb_params,
                   dtrain,
                   nfold=5,
                   num_boost_round=200,
                   early_stopping_rounds=5,
                   verbose_eval=10,
                   show_stdv=False
                   )
num_boost_rounds = len(cv_result)
print(num_boost_rounds)
# train model
print("Training model ...")
model = xgb.train(dict(xgb_params, silent=1), dtrain,
                  num_boost_round=num_boost_rounds)

print("Predict eval data ...")                  
pred = model.predict(dtest)
y_pred = []

for i, predict in enumerate(pred):
    y_pred.append(str(round(predict, 4)))
y_pred = np.array(y_pred)

print("Export submission ...")                  
output = pd.DataFrame({'ParcelId': properties['parcelid'].astype(np.int32),
                       '201610': y_pred, '201611': y_pred, '201612': y_pred,
                       '201710': y_pred, '201711': y_pred, '201712': y_pred})
# set col 'ParceID' to first col
cols = output.columns.tolist()
cols = cols[-1:] + cols[:-1]
output = output[cols]
output.to_csv(DATA_DIR + '/sub{}.csv'.format(
    datetime.now().strftime('%Y%m%d_%H%M%S')), index=False)
