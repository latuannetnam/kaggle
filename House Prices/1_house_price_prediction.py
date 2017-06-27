# Predict house price.
# House Prices: Advanced Regression Techniques competition:
# https://www.kaggle.com/c/house-prices-advanced-regression-techniques/kernels?userId=1075113
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split, validation_curve, learning_curve
from sklearn.metrics import mean_squared_error, make_scorer
from xgboost import XGBRegressor
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


def prepare_data():
    X = train_data[features]
    # Label data: using logarithm to transform skewed data
    #  -> increase ML efficiency
    Y = np.log(train_data[label])
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.33, random_state=324)
    print("train size:", len(x_train))
    print("test size:", len(x_test))
    print("Split ratio", len(x_test) / len(x_train))
    return X, Y, x_train, x_test, y_train, y_test


def linear_model():
    # RMSE: 0.206124028461
    # Submission score: 0.22756
    model = LinearRegression()
    model.fit(X_train, Y_train)
    return 'LinearRegression', model


def decision_tree_model():
    # RMSE: 0.303670257561
    model = DecisionTreeRegressor(max_depth=20)
    model.fit(X_train, Y_train)
    return 'DecisionTree', model


def random_forest_model():
    # RMSE1: 0.220141683969
    # RMSE2: 0.18198 => added Year build
    # Your submission scored 0.18198
    # Rank: 1:1801 => 2:1656
    model = RandomForestRegressor(n_estimators=500)
    model.fit(X_train, Y_train)
    return 'RandomForest', model


def xgboost_model():
    # RMSE: 0.2103118478317821
    # RMSE2: 0.18213109708173356
    # Your submission scored 0.21315, which is an improvement ofyour previous score of 0.21609. Great job!
    # Rank: 1794
    model = XGBRegressor(max_depth=3, n_estimators=100)
    model.fit(X_train, Y_train)
    return 'XGBoost', model


def deepNN_model():
    # RMSE: 0.21037404583781194
    # RMSE: 0.216011400876183
    # Your submission scored 0.22743
    # Rank: 1801
    NUM_LAYERS = 4
    NUM_HIDDEN_NODES = 256
    MINI_BATCH_SIZE = 10
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.003
    TRAIN_SPLIT = 1.

    model = NNRegressor(X_train, Y_train,
                        NUM_LAYERS, NUM_HIDDEN_NODES,
                        LEARNING_RATE,
                        NUM_EPOCHS, MINI_BATCH_SIZE,
                        TRAIN_SPLIT)
    model.dump_input()
    model.fit()
    model.plot(X_train.values.astype(float), np.reshape(
        Y_train.values.astype(float), (-1, 1)))
    model.plot(X_test.values.astype(float), np.reshape(
        Y_test.values.astype(float), (-1, 1)))
    return 'DeepNN', model


def rmse(model):
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    RMSE2 = np.sqrt(-cross_val_score(model, X_test,
                                     Y_test, scoring=scorer, cv=10))
    return RMSE2.mean()


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


def export_saleprice():
    X_eval = eval_data[features]
    isnull_data = X_eval.isnull().any()
    print(isnull_data[isnull_data == True].sort_index())
    # filling Null daa
    X_eval['GarageArea'].fillna(X_eval['GarageArea'].mean(), inplace=True)
    X_eval['TotalBsmtSF'].fillna(X_eval['TotalBsmtSF'].mean(), inplace=True)
    # X_eval['GarageArea'][:5]
    print("Predict SalePrice for test data. Use numpy.exp to transform SalePrice to normal (model predicts on log of SalePrice)")
    # Uncomment if using with TensorFlow
    # Y_eval_log = model.predict(X_eval.values.astype(float))
    Y_eval_log = model.predict(X_eval)
    # Transform SalePrice to normal
    Y_eval = np.exp(Y_eval_log.ravel())
    print(Y_eval[:5])
    # save predicted sale price to CSV
    eval_output = pd.DataFrame({'Id': eval_data['Id'], 'SalePrice': Y_eval})
    print("Evaluation output len:", len(eval_output))
    today = str(datetime.date.today())
    eval_output.to_csv(DATA_DIR + '/' + today + '-' +
                       model_name + '.csv', index=False)


# ---- Main program --------------
# Load data. Download
# from:https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
train_data = pd.read_csv(DATA_DIR + "/train.csv")
eval_data = pd.read_csv(DATA_DIR + "/test.csv")
label = 'SalePrice'
features = ['GrLivArea', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'YearBuild']
# exploring train data
explore_data()
train_data = clean_data(train_data)
# prepare data for modeling
X, Y, X_train, X_test, Y_train, Y_test = prepare_data()

print("Model building ......")
# model_name, model = linear_model()
# model_name, model = decision_tree_model()
# model_name, model = random_forest_model()
# model_name, model = deepNN_model()
model_name, model = xgboost_model()

print("Predict Sale price to training data using:", model_name)

# Uncomment if using with TensorFlow
# Y_prediction = model.predict(X_test.values.astype(float))
Y_prediction = model.predict(X_test)
print(Y_prediction[:5])
# RMSE = rmse(model))
RMSE = sqrt(mean_squared_error(y_true=Y_test, y_pred=Y_prediction))
print("RMSE:", RMSE)
# plot prediction data
plot_prediction()

# plot learning curve
plot_learning_curve()
# Export predicted sale price of evaluation data to submit to competition
export_saleprice()
