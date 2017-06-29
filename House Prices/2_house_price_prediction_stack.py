# Predict house price. Using stacked model.
# House Prices: Advanced Regression Techniques competition:
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques/kernels?userId=1075113
# Credit:
# https://dnc1994.com/2016/05/rank-10-percent-in-first-kaggle-competition-en/

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score, train_test_split, learning_curve, validation_curve, KFold
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel
from xgboost.sklearn import XGBRegressor
from math import sqrt
from scipy import stats
import datetime
import sys
from inspect import getsourcefile
import os.path
# sys.path.insert(0, '..')
current_path = os.path.abspath(getsourcefile(lambda: 0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]
sys.path.insert(0, parent_dir)
model_dir = parent_dir + "/model"
sys.path.insert(0, model_dir)
print("parent dir:", parent_dir)

import NNRegressor
from NNRegressor import NNRegressor

# Input data files are available in the DATA_DIR directory.
DATA_DIR = "data-temp"

# Filter out feature with score <= threshold in percent
IMPORTANT_THRESHOLD_PERCENT = 20

# Filter out feature with score <= threshold
IMPORTANT_THRESHOLD = 0.00

# Filter out feature with NaN percent > threshold
NAN_THRESHOLD = 0.9
# number of kfolds
N_FOLDS = 10


def explore_data():
    print("Explore data ....")
    print(train_data.head(5))
    train_data_columns = train_data.columns.values
    eval_data_columns = eval_data.columns.values
    print("Train data size:", len(train_data))
    print("Eval data size:", len(eval_data))
    print("Missing columns in eval data:", np.setdiff1d(
        train_data_columns, eval_data_columns))


def clean_data(train_data):
    print("Cleaning data ....")
    # GrLivArea
    train_data = train_data.drop(train_data[train_data['Id'] == 1299].index)
    train_data = train_data.drop(train_data[train_data['Id'] == 524].index)
    # GarageArea
    train_data = train_data.drop(train_data[train_data['Id'] == 582].index)
    train_data = train_data.drop(train_data[train_data['Id'] == 1191].index)
    train_data = train_data.drop(train_data[train_data['Id'] == 1062].index)
    # TotalBsmtSF
    train_data = train_data.drop(train_data[train_data['Id'] == 333].index)
    train_data = train_data.drop(train_data[train_data['Id'] == 497].index)
    train_data = train_data.drop(train_data[train_data['Id'] == 441].index)

    # 1stFlrSF
    train_data = train_data.drop(train_data[train_data['Id'] == 1025].index)
    return train_data


def transform_data(train_data, eval_data, features):
    # Seperate input and label from train_data
    input_data = train_data[features]

    # Combine train + eval data
    combine_data = pd.concat([input_data, eval_data], keys=['train', 'eval'])
    combine_data.head(5)

    # Get high percent of NaN data
    null_data = combine_data.isnull()
    total = null_data.sum().sort_values(ascending=False)
    percent = (null_data.sum() / null_data.count()
               ).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1,
                             keys=['Total', 'Percent'])
    high_percent_miss_data = missing_data[missing_data['Percent']
                                          > NAN_THRESHOLD]
    print("high percent missing:", high_percent_miss_data)
    # Drop high percent NaN columns
    drop_columns = high_percent_miss_data.index.values
    print("Drop columns:", drop_columns)
    combine_data.drop(drop_columns, axis=1, inplace=True)
    print("Number of features:", len(combine_data.columns))

    print("Filling NaN values ...")
    # Get data with column type is number
    numeric_data = combine_data.select_dtypes(include=[np.number])
    # For each NaN, fill with mean of columns'value
    null_data = numeric_data.isnull().sum().sort_values(ascending=False)
    # print("Before fillNaN")
    # print(null_data[null_data>0])
    cols = numeric_data.columns.values
    numeric_features = cols
    for col in cols:
        combine_data[col].fillna(combine_data[col].mean(), inplace=True)

    # Get data with column type is number
    object_data = combine_data.select_dtypes(include=['object'])
    # For each NaN, fill with mean of columns'value
    null_data = object_data.isnull().sum().sort_values(ascending=False)
    # print("Before fillNaN")
    # print(null_data[null_data>0])
    cols = object_data.columns.values
    object_features = cols
    for col in cols:
        combine_data[col].fillna(combine_data[col].mode()[0], inplace=True)

    null_data = combine_data.isnull().sum().sort_values(ascending=False)
    print("After fillNaN")
    print(null_data[null_data > 0])

    print("Feature scaling ...")
    # Standarlize object data with Label Encoder
    object_data_standardlized = combine_data[object_features].apply(
        LabelEncoder().fit_transform)
    combine_data.update(object_data_standardlized)

    std_scale = StandardScaler().fit_transform(combine_data.astype(float).values)
    combine_data_scale = pd.DataFrame(
        std_scale, index=combine_data.index, columns=combine_data.columns)

    # split train_set and evaluation set
    train_set = combine_data_scale.loc['train']
    eval_set = combine_data_scale.loc['eval']

    return train_set, eval_set


def split_train_set(train_set, target):
    X = train_set
    # Label data: using logarithm to transform skewed data
    #  -> increase ML efficiency
    Y = target
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.31, random_state=324)
    print("train size:", len(x_train))
    print("test size:", len(x_test))
    print("Split ratio", len(x_test) / len(x_train))
    return X, Y, x_train, x_test, y_train, y_test


def linear_model():
    # RMSE: 0.206124028461
    # Submission score: 0.22756
    model = LinearRegression(n_jobs=-1)
    # model.fit(X_train, Y_train)
    return 'LinearRegression', model


def decision_tree_model():
    # RMSE: 0.303670257561
    model = DecisionTreeRegressor(max_depth=20)
    # model.fit(X_train, Y_train)
    return 'DecisionTree', model


def random_forest_model():
    # RMSE1: 0.220141683969
    # RMSE2: 0.18198 => added Year build
    # Your submission scored 0.18198
    # Rank: 1:1801 => 2:1656
    model = RandomForestRegressor(n_estimators=500, n_jobs=-1)
    # model.fit(X_train, Y_train)
    return 'RandomForest', model


def xgboost_model():
    # RMSE: 0.2103118478317821
    # RMSE2: 0.18213109708173356
    # Your submission scored 0.21315, which is an improvement ofyour previous score of 0.21609. Great job!
    # Rank: 1794
    # model = XGBRegressor(max_depth=3, n_estimators=100)
    model = XGBRegressor(n_estimators=500, max_depth=5, n_jobs=-1)
    # model.fit(X_train, Y_train)
    return 'XGBoost', model


def gbm_model():
    model = GradientBoostingRegressor(n_estimators=500)
    return 'GradientBoost', model


def extra_tree_model():
    model = ExtraTreesRegressor(n_estimators=500, n_jobs=-1)
    return "ExtraTreeBoost", model


def deepNN_model(x, y):
    # RMSE: 0.21037404583781194
    # RMSE: 0.216011400876183
    # Your submission scored 0.22743
    # Rank: 1801
    NUM_LAYERS = 4
    NUM_HIDDEN_NODES = 256
    MINI_BATCH_SIZE = 10
    NUM_EPOCHS = 10000
    LEARNING_RATE = 0.1
    TRAIN_SPLIT = 1.

    model = NNRegressor(x, y,
                        NUM_LAYERS, NUM_HIDDEN_NODES,
                        LEARNING_RATE,
                        NUM_EPOCHS, MINI_BATCH_SIZE,
                        TRAIN_SPLIT)
    model.dump_input()
    model.fit()
    # model.plot(x.values.astype(float), np.reshape(
    #     y.values.astype(float), (-1, 1)))
    return 'DeepNN', model


def rmse(y_true, y_prediction):
    return sqrt(mean_squared_error(y_true=y_true, y_pred=y_prediction))


def rmse2(model):
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    RMSE2 = np.sqrt(-cross_val_score(model, X_test,
                                     Y_test, scoring=scorer, cv=10))
    return RMSE2.mean()


def importance_features(model, x, y, threshold):
    model.fit(x, y)
    features_score = pd.Series(
        model.feature_importances_, index=x.columns.values)
    print("Feature importance:", features_score.describe())
    # Drop features with score below threshold
    features = features_score[features_score > threshold].index
    # features = features_score[features_score >=
    #                           np.percentile(features_score, threshold)].index
    print("Remain features:", len(features))
    return features


def build_model(model_name, model, boost=False):
    # features = importance_features(
    #     model, train_set, target_log, IMPORTANT_THRESHOLD)
    features = train_set.columns.values
    print("Model:", model_name, "Importance features:", len(features))
    # print(features)
    model_dict = {
        'model_name': model_name,
        'model': model,
        'features': features
    }
    if (boost):
        model_boost = AdaBoostRegressor(
            base_estimator=model, n_estimators=200, random_state=200)
        model_dict['model'] = model_boost

    return model_dict


def build_models_level1():
    # model_name, model = linear_model()
    # model_name, model = decision_tree_model()
    # model_name, model = random_forest_model()
    # model_name, model = deepNN_model()
    models = []

    # Extra Tree Boostt
    model_name, model = extra_tree_model()
    model_dict = build_model(model_name, model)
    models.append(model_dict.copy())

    # XGBoost
    model_name, model = xgboost_model()
    model_dict = build_model(model_name, model)
    models.append(model_dict.copy())

    # Random forest
    model_name, model = random_forest_model()
    model_dict = build_model(model_name, model)
    models.append(model_dict.copy())

    # Decision Tree
    model_name, model = decision_tree_model()
    model_dict = build_model(model_name, model, True)
    models.append(model_dict.copy())

    # Linear decision
    model_name, model = linear_model()
    model_dict = build_model(model_name, model, True)
    models.append(model_dict.copy())

    # Gradient Boost
    model_name, model = gbm_model()
    model_dict = build_model(model_name, model)
    models.append(model_dict.copy())
    return models


# Kfold, train for each model, stack result
def model_stack_train(models, X, Y, T):
    kfolds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=321)
    S_train = np.zeros((X.shape[0], len(models)))
    S_test = np.zeros((T.shape[0], len(models)))
    print("S_train shape:", S_train.shape, " S_test shape:",
          S_test.shape)
    for i, model_dict in enumerate(models):
        model_name = model_dict['model_name']
        model = model_dict['model']
        features = model_dict['features']
        print("Base model:", model_name)
        S_test_i = np.zeros((T.shape[0], N_FOLDS))
        total_rmse = 0
        for j, (train_idx, test_idx) in enumerate(kfolds.split(X)):
            X_train = X[train_idx]
            y_train = Y[train_idx]
            X_holdout = X[test_idx]
            y_holdout = Y[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_holdout)[:]
            S_train[test_idx, i] = y_pred
            S_test_i[:, j] = model.predict(T)[:]
            rmse1 = rmse(y_holdout, y_pred)
            total_rmse = total_rmse + rmse1
            print("fold:", j + 1, "rmse:", rmse1)

        S_test[:, i] = S_test_i.mean(1)
        print("Avg rmse:", total_rmse / (j + 1))

    print("Detect zero value")
    print(np.where(S_train == 0))
    print(np.where(S_test == 0))
    return S_train, S_test


# train and predict with given model, X: input, Y:label, T: test set
def model_train_predict(model, X, Y, T):
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.31, random_state=324)
    print("Trainning ...")
    model.fit(x_train, y_train)
    y_test_pred = model.predict(x_test)
    print("rmse:", rmse(y_test, y_test_pred))
    y_pred = model.predict(T)[:]
    return y_pred


# Kfold, Fit and predict for model
def kfold_fit_predict(model, X, Y, T):
    kfolds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=321)
    total_rmse = 0
    for j, (train_idx, test_idx) in enumerate(kfolds.split(X)):
        X_train = X[train_idx]
        y_train = Y[train_idx]
        X_holdout = X[test_idx]
        y_holdout = Y[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_holdout)[:]
        rmse1 = rmse(y_holdout, y_pred)
        total_rmse = total_rmse + rmse1
        print("fold:", j, "rmse:", rmse1)
    print("Avg rmse:", total_rmse / (j + 1))

    print("Predict final value")
    y_pred = model.predict(T)[:]
    return y_pred


def model_stacking_avg():
    print("Model stacking using AVG....")
    print("Round 2: Predicting for evaluation data using model stacking ...")
    model_eval_predictions['mean_predict'] = model_eval_predictions.mean(
        axis=1)
    print(model_eval_predictions[:5])
    return model_eval_predictions['mean_predict'].values


def model_stacking_boost():
    # GDM(n=100, depth=1, loss='huber')=> stack (XB + RF + LR(boost) + DT(boost) + GBM + ET(boost), 10 kfolds)
    # rmse: 0.1392549, LB score:
    print("Model stacking using GBM....")
    model = GradientBoostingRegressor(n_estimators=100, max_depth=1, loss='huber')
    model_boost = AdaBoostRegressor(
        base_estimator=model, n_estimators=200, random_state=200)
    # features = importance_features(model, X, Y, IMPORTANT_THRESHOLD)
    features = X.columns.values
    X_in = X[features].values
    Y_in = Y.values
    T_in = eval_set[features].values
    X_out, T_out = model_stack_train(models, X_in, Y_in, T_in)

    # find best param
    # 'min_child_weight':  [1, 2, 5, 10]
    # print("Finding best param for model ....")
    # param_grid = {"n_estimators": [50, 100, 200, 500],
    #               "max_depth": [1, 5, 10, 50],
    #               "loss": ['ls', 'lad', 'huber', 'quantile']
    #               }
    # grid_search = GridSearchCV(model, param_grid, n_jobs=1, cv=5)
    # grid_search.fit(X_out, Y_in)
    # print(grid_search.best_params_)
    # quit()

    print("Predict final value with stacking model")
    y_prediction = model_train_predict(model, X_out, Y_in, T_out)
    print(y_prediction[:5])
    return y_prediction


def model_stacking_xgboost():
    # AdaBoost(XGB)=> stack (ZXB + RF + LR(boost), 5 kfolds)
    # rmse:0.16098025986999429, LB score: 0.14577
    # AdaBoost(XGB)=> stack (ZXB + RF + LR(boost), 10 kfolds)
    # rmse:0.16145689394773224, LB score: 0.13951
    # AdaBoost(XGB)=> stack (ZXB + RF + LR(boost) + GBM, 10 kfolds)
    # rmse:0.1555152075958718, LB score:
    # AdaBoost(XGB)=> stack (ZXB + GBM, 10 kfolds)
    # rmse:0.16145230381256545, LB score:
    # AdaBoost(XGB)=> stack (ZXB + RF + LR(boost) + DT + GBM, 10 kfolds)
    # rmse:0.15018237407328916, LB score: 0.13643
    # XBoost(n=200, depth=1)=> stack (XB + RF + LR(boost) + DT + GBM, 10 kfolds)
    # rmse: 0.14612253864, LB score: 0.13643
    # XBoost(n=200, depth=1)=> stack (XB + RF + LR(boost) + DT(boost) + GBM, 10 kfolds)
    # rmse: 0.14449918, LB score: 0.13643
    # XBoost(n=200, depth=1)=> stack (XB + RF + DT(boost) + GBM, 10 kfolds)
    # rmse: 0.14486038, LB score: 0.13643
    # XBoost(n=200, depth=1)=> stack (XB + RF + DT(boost) + GBM + ET(boost), 10 kfolds)
    # rmse:  0.14513938, LB score: 0.13643
    # XBoost(n=200, depth=1)=> stack (XB + RF + LR(boost) + DT(boost) + GBM + ET(boost), 10 kfolds)
    # rmse:  0.145002692, LB score: 0.13643

    print("Model stacking using xgboosting....")
    model = XGBRegressor(n_estimators=200, max_depth=1, n_jobs=-1)
    model_boost = AdaBoostRegressor(
        base_estimator=model, n_estimators=200, random_state=200)
    # features = importance_features(model, X, Y, IMPORTANT_THRESHOLD)
    features = X.columns.values
    X_in = X[features].values
    Y_in = Y.values
    T_in = eval_set[features].values
    X_out, T_out = model_stack_train(models, X_in, Y_in, T_in)

    # find best param
    # 'min_child_weight':  [1, 2, 5, 10]
    # print("Finding best param for model ....")
    param_grid = {"n_estimators": [50, 100, 200, 500],
                  "max_depth": [1, 5, 10, 50],
                  }
    # grid_search = GridSearchCV(model, param_grid, n_jobs=1, cv=5)
    # grid_search.fit(X_out, Y_in)
    # print(grid_search.best_params_)

    print("Predict final value with stacking model")
    y_prediction = model_train_predict(model, X_out, Y_in, T_out)
    print(y_prediction[:5])
    return y_prediction


def model_stacking_deepNN():
    print("Model stacking using DeepNN  ....")
    model_name, model = deepNN_model(model_train_predictions, Y_train)
    y_prediction = model.predict(model_test_predictions)
    print(model_name, ":", rmse(Y_test, y_prediction))
    model.plot(model_predictions.values.astype(float), np.reshape(
        Y_test.values.astype(float), (-1, 1)))

    print("Round 2: Predicting for evaluation data using model stacking ...")
    y_prediction = model.predict(model_eval_predictions)
    print(y_prediction[:5])
    return y_prediction


def model_kfold_xgboost():
    # rmse: 0.12772005156128946
    # Best score: 0.13676
    # rmse: 0.12225599896729336
    # Best score: 0.13659

    print("Model stacking using xgboosting....")
    model_name, model = xgboost_model()
    model_boost = AdaBoostRegressor(
        base_estimator=model, n_estimators=200, random_state=200)
    # model_boost.fit(x_train, y_train)
    features = importance_features(model, X, Y, IMPORTANT_THRESHOLD)
    X_in = X[features].values
    Y_in = Y.values
    T = eval_set[features].values

    y_prediction = kfold_fit_predict(model_boost, X_in, Y_in, T)
    print(y_prediction[:5])
    return y_prediction


def plot_prediction():
    plt.scatter(Y_prediction, Y_test, c="b",
                marker="s", label="Validation data")
    plt.title(model_name)
    plt.xlabel("Predicted values")
    plt.ylabel("Real values")
    plt.legend(loc="upper left")
    plt.show()


def plot_learning_curve():
    plt.figure()
    plt.title('Learning curve:' + model_name)
    # if ylim is not None:
    #     plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(model, X, Y)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()


def export_saleprice(y_prediction):
    # Transform SalePrice to normal
    Y_eval = np.exp(y_prediction.ravel())
    print(Y_eval[:5])
    # save predicted sale price to CSV
    eval_output = pd.DataFrame({'Id': eval_data['Id'], 'SalePrice': Y_eval})
    print("Evaluation output len:", len(eval_output))
    today = str(datetime.date.today())
    eval_output.to_csv(DATA_DIR + '/' + today + '-' +
                       'stack' + '.csv', index=False)


# ---- Main program --------------
# Load data. Download
# from:https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
train_data = pd.read_csv(DATA_DIR + "/train.csv")
eval_data = pd.read_csv(DATA_DIR + "/test.csv")
# exploring train data
explore_data()

# Label and Ground trust
label = 'SalePrice'
target = train_data[label]
target_log = np.log(target)

# Store eval_id
eval_data_id = eval_data['Id']

# Features
eval_data_columns = eval_data.columns.values
features = eval_data_columns
features = np.setdiff1d(features, ['Id'])
print("Features:", len(features))

# transform data: fillNa, feature scaling
train_set, eval_set = transform_data(train_data, eval_data, features)
print(train_set.iloc[:5, :5])

# split train set data for modeling
X, Y, X_train, X_test, Y_train, Y_test = split_train_set(
    train_set, target_log)

# print("Model building ......")
models = build_models_level1
# print("Training model ...")
# model_train_predictions, model_test_predictions = train_models(models)
# print(model_test_predictions.head(5))
# print("Round 1:Predicting for evaluation data")
# model_eval_predictions = predict_eval_set(models, eval_set)
# print(model_eval_predictions.head(5))

# Use DeepNNModel to stack models
# y_prediction = model_stacking_xgboost()
y_prediction = model_stacking_boost()

# y_prediction = model_kfold_xgboost()

# Export predicted sale price of evaluation data to submit to competition
export_saleprice(y_prediction)
