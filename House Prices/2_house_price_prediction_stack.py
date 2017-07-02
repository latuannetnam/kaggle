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
# NAN_THRESHOLD = 0.9
NAN_THRESHOLD = 0
# number of kfolds
N_FOLDS = 5

# Level name
LEVEL_1 = 'level_1'
LEVEL_2 = 'level_2'
LEVEL_3 = 'level_3'


class StackRegression:
    def __init__(self, label):
        self.label = label

    def load_data(self):
        # Load data. Download
        # from:https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data
        self.train_data = pd.read_csv(DATA_DIR + "/train.csv")
        self.eval_data = pd.read_csv(DATA_DIR + "/test.csv")

        # Label and Ground trust
        self.target = self.train_data[self.label]
        self.target_log = np.log(self.target)

        # Store eval_id
        self.eval_data_id = self.eval_data['Id']
        train_data_columns = self.train_data.columns.values
        eval_data_columns = self.eval_data.columns.values
        print("Train data size:", len(self.train_data))
        print("Eval data size:", len(self.eval_data))
        print("Missing columns in eval data:", np.setdiff1d(
            train_data_columns, eval_data_columns))

        # Features
        eval_data_columns = self.eval_data.columns.values
        features = eval_data_columns
        self.features = np.setdiff1d(features, ['Id'])
        print("Features:", len(self.features))

    def explore_data(self):
        print("Explore data ....")
        print(self.train_data.head(5))

    def clean_data(self):
        print("Cleaning data ....")
        # GrLivArea
        self.train_data = self.train_data.drop(
            self.train_data[self.train_data['Id'] == 1299].index)
        self.train_data = self.train_data.drop(
            self.train_data[self.train_data['Id'] == 524].index)
        # GarageArea
        self.train_data = self.train_data.drop(
            self.train_data[self.train_data['Id'] == 582].index)
        self.train_data = self.train_data.drop(
            self.train_data[self.train_data['Id'] == 1191].index)
        self.train_data = self.train_data.drop(
            self.train_data[self.train_data['Id'] == 1062].index)
        # TotalBsmtSF
        self.train_data = self.train_data.drop(
            self.train_data[self.train_data['Id'] == 333].index)
        self.train_data = self.train_data.drop(
            self.train_data[self.train_data['Id'] == 497].index)
        self.train_data = self.train_data.drop(
            self.train_data[self.train_data['Id'] == 441].index)

        # 1stFlrSF
        self.train_data = self.train_data.drop(
            self.train_data[self.train_data['Id'] == 1025].index)

    def fill_null_values(self, data):
        # credit to:Tanner Carbonati
        # (https://www.kaggle.com/tannercarbonati/detailed-data-analysis-ensemble-modeling)

        # Get high percent of NaN data
        null_data = data.isnull()
        total = null_data.sum().sort_values(ascending=False)
        percent = (null_data.sum() / null_data.count()
                   ).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1,
                                 keys=['Total', 'Percent'])
        high_percent_miss_data = missing_data[missing_data['Percent']
                                              > NAN_THRESHOLD]
        print("high percent missing:", len(high_percent_miss_data))
        # Drop high percent NaN columns
        drop_columns = high_percent_miss_data.index.values
        # print("Drop columns:", drop_columns)
        # data.drop(drop_columns, axis=1, inplace=True)
        print("Filling NaN values ...")
        # Get data with column type is number
        numeric_data = data.select_dtypes(include=[np.number])
        # For each NaN, fill with mean of columns'value
        cols = numeric_data.columns.values
        numeric_features = cols
        for col in cols:
            data[col].fillna(data[col].mean(), inplace=True)

        # Get data with column type is object
        object_data = data.select_dtypes(include=['object'])
        cols = object_data.columns.values
        object_features = cols
        print('Object columns', len(cols))
        object_data = data[cols].fillna(value='None')
        data.update(object_data)
        print(object_data['Alley'].head(5))
        # For each NaN, fill with mean of columns'value
        # for col in cols:
        #     data[col].fillna(data[col].mode()[0], inplace=True)

        null_data = data.isnull().sum().sort_values(ascending=False)
        print("After fillNaN")
        print(null_data[null_data > 0])
        print("Number of features:", len(data.columns))

        return object_features

    def transform_data(self):
        # Seperate input and label from train_data
        input_data = self.train_data[self.features]

        # Combine train + eval data
        combine_data = pd.concat(
            [input_data, self.eval_data], keys=['train', 'eval'])
        object_features = self.fill_null_values(combine_data)
        # print(combine_data['Alley'].head(5))
        print("Feature scaling ...")
        # Standarlize object data with Label Encoder
        object_data_standardlized = combine_data[object_features].apply(
            LabelEncoder().fit_transform)
        combine_data.update(object_data_standardlized)
        # Transform numerial data
        std_scale = StandardScaler().fit_transform(combine_data.astype(float).values)
        combine_data_scale = pd.DataFrame(
            std_scale, index=combine_data.index, columns=combine_data.columns)

        # split train_set and evaluation set
        self.train_set = combine_data.loc['train']
        self.eval_set = combine_data.loc['eval']
        self.train_set_scale = combine_data_scale.loc['train']
        self.eval_set_scale = combine_data_scale.loc['eval']
        print("Before transform:")
        print(self.train_set.iloc[:5, :5])
        print("After transform:")
        print(self.train_set_scale.iloc[:5, :5])

    def deepNN_model(self, x, y):
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

    def rmse(self, y_true, y_prediction):
        return sqrt(mean_squared_error(y_true=y_true, y_pred=y_prediction))

    def rmse2(self, model):
        scorer = make_scorer(mean_squared_error, greater_is_better=False)
        RMSE2 = np.sqrt(-cross_val_score(model, X_test,
                                         Y_test, scoring=scorer, cv=10))
        return RMSE2.mean()

    def importance_features(self, model, x, y, threshold):
        model.fit(x, y)
        features_score = pd.Series(
            model.feature_importances_, index=x.columns.values)
        print("Feature importance:", features_score.describe())
        # Drop features with score below threshold
        features = features_score[features_score > threshold].index
        # features = features_score[features_score >=
        # np.percentile(features_score, threshold)].index
        print("Remain features:", len(features))
        return features

    def build_models_level1(self):
        print('Bulding models level 1..')
        models = []
        # model_name = model.__class__.__name__
        # XGBoost
        model_dict = {
            # 'model': XGBRegressor(n_estimators=500, max_depth=5, n_jobs=-1),
            'model': XGBRegressor(n_estimators=500, max_depth=3, n_jobs=-1, random_state=123),
            'model_param': {"n_estimators": [50, 100, 200, 500],
                            "max_depth": [1, 3, 5, 10],
                            },
            'boost': False
        }
        models.append(model_dict.copy())

        # Extra Tree Boostt
        model_dict = {
            # 'model': ExtraTreesRegressor(n_estimators=500, n_jobs=-1),
            'model': ExtraTreesRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=456),
            'model_param': {"n_estimators": [50, 100, 200, 500],
                            "max_depth": [1, 3, 5, 10],
                            },
            'boost': False
        }
        models.append(model_dict.copy())

        # Random forest
        model_dict = {
            # 'model': RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=200),
            'model': RandomForestRegressor(n_estimators=500, max_depth=10, n_jobs=-1, random_state=789),
            'model_param': {"n_estimators": [50, 100, 200, 500],
                            "max_depth": [1, 3, 5, 10],
                            },
            'boost': False
        }
        models.append(model_dict.copy())

        # Decision Tree
        model_dict = {
            'model': DecisionTreeRegressor(max_depth=10, random_state=146),
            'model_param': {"max_depth": [1, 10, 20, 50],
                            },
            'boost': True
        }
        models.append(model_dict.copy())

        # Gradient Boost
        model_dict = {
            'model': GradientBoostingRegressor(n_estimators=500, max_depth=1, random_state=357),
            'model_param': {"n_estimators": [50, 100, 200, 500],
                            "max_depth": [1, 10, 20, 50],
                            },
            'boost': True
        }
        models.append(model_dict.copy())
        print('Total model:', len(models))
        return models

    def build_models_level2(self):
        print('Bulding models level 2 ..')
        models = []
        # model_name = model.__class__.__name__
        # XGBoost
        model_dict = {
            # 'model': XGBRegressor(n_estimators=500, max_depth=5, n_jobs=-1),
            'model': XGBRegressor(n_estimators=100, max_depth=1, n_jobs=-1, random_state=123),
            'model_param': {"n_estimators": [50, 100, 200, 500],
                            "max_depth": [1, 3, 5, 10],
                            },
            'boost': False
        }
        models.append(model_dict.copy())

        # Extra Tree Boostt
        model_dict = {
            # 'model': ExtraTreesRegressor(n_estimators=500, n_jobs=-1),
            'model': ExtraTreesRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=456),
            'model_param': {"n_estimators": [50, 100, 200, 500],
                            "max_depth": [1, 3, 5, 10],
                            },
            'boost': False
        }
        # models.append(model_dict.copy())

        # Random forest
        model_dict = {
            # 'model': RandomForestRegressor(n_estimators=500, n_jobs=-1, random_state=200),
            'model': RandomForestRegressor(n_estimators=500, max_depth=10, n_jobs=-1, random_state=789),
            'model_param': {"n_estimators": [50, 100, 200, 500],
                            "max_depth": [1, 3, 5, 10],
                            },
            'boost': False
        }
        # models.append(model_dict.copy())

        # Decision Tree
        model_dict = {
            'model': DecisionTreeRegressor(max_depth=10, random_state=146),
            'model_param': {"max_depth": [1, 10, 20, 50],
                            },
            'boost': True
        }
        # models.append(model_dict.copy())

        # Gradient Boost
        model_dict = {
            'model': GradientBoostingRegressor(n_estimators=200, max_depth=1, random_state=357),
            'model_param': {"n_estimators": [50, 100, 200, 500],
                            "max_depth": [1, 10, 20, 50],
                            },
            'boost': True
        }
        models.append(model_dict.copy())
        print('Total model:', len(models))
        return models

    def search_best_params(self, models, X, Y):
        for model_dict in models:
            model = model_dict['model']
            model_name = model.__class__.__name__
            print('Searching best param for model:', model_name)
            param_grid = model_dict['model_param']
            grid_search = GridSearchCV(model, param_grid, n_jobs=1, cv=5)
            grid_search.fit(X, Y)
            print(grid_search.best_params_)
        quit()

    # Kfold, train for each model, stack result
    def model_stack_train(self, models, X_in, Y_in, T_in):
        n_folds = len(models)
        kfolds = KFold(n_splits=n_folds, shuffle=True, random_state=321)
        S_train = np.zeros((X_in.shape[0], len(models)))
        S_test = np.zeros((T_in.shape[0], len(models)))
        print("S_train shape:", S_train.shape, " S_test shape:",
              S_test.shape)
        all_rmse = 0
        for i, model_dict in enumerate(models):
            model_temp = model_dict['model']
            boost = model_dict['boost']
            model_name = model_temp.__class__.__name__
            print("Base model:", model_name, " boost:", boost)
            if boost:
                model = AdaBoostRegressor(
                    base_estimator=model_temp, n_estimators=200, random_state=200)
            else:
                model = model_temp
            S_test_i = np.zeros((T_in.shape[0], n_folds))
            model_rmse = 0
            for j, (train_idx, test_idx) in enumerate(kfolds.split(X_in)):
                X_train = X_in[train_idx]
                y_train = Y_in[train_idx]
                X_holdout = X_in[test_idx]
                y_holdout = Y_in[test_idx]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred
                S_test_i[:, j] = model.predict(T_in)[:]
                rmse1 = self.rmse(y_holdout, y_pred)
                model_rmse = model_rmse + rmse1
                all_rmse = all_rmse + rmse1
                print("fold:", j + 1, "rmse:", rmse1)

            S_test[:, i] = S_test_i.mean(1)
            print("Model rmse:", model_rmse / (j + 1))
        print("All AVG rmse:", all_rmse / (j + 1) / len(models))
        # print("Detect zero value")
        # print(np.where(S_train == 0))
        # print(np.where(S_test == 0))
        return S_train, S_test

    def save_train_data(self, level, X_in, Y_in, T_in):
        save_dir = DATA_DIR + "/" + level
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        d_train = pd.DataFrame(data=X_in)
        d_train['label'] = Y_in
        d_train.to_csv(save_dir + '/train.csv', index=False)
        print("predicted from train data level:", level)
        print(d_train.head(5))
        d_test = pd.DataFrame(data=T_in)
        d_test['Id'] = self.eval_data_id
        print("predicted from eval data level:", level)
        print(d_test.head(5))
        d_test.to_csv(save_dir + '/test.csv', index=False)

    def load_pretrained_data(self, level):
        print('Load train data for level: ', level)
        save_dir = DATA_DIR + "/" + level
        train_set = pd.read_csv(save_dir + "/train.csv")
        eval_set = pd.read_csv(save_dir + "/test.csv")
        Y_in = train_set['label'].values
        X_in = train_set.drop(['label'], axis=1).values
        print("Train data for level:", level)
        print(X_in[:5])
        self.eval_data_id = eval_set['Id'].values
        T_in = eval_set.drop(['Id'], axis=1).values
        print("Eval data for level:", level)
        print(T_in[:5])
        return X_in, Y_in, T_in

    def train_level1_single_dataset(self, X, Y, T):
        features = X.columns.values
        X_in = X[features].values
        Y_in = Y.values
        T_in = T[features].values
        models_level1 = self.build_models_level1()
        # self.search_best_params(models, X, Y)
        X_out, T_out = self.model_stack_train(models_level1, X_in, Y_in, T_in)
        # print("X")
        # print(X.iloc[:5, :5])
        # print("X_out")
        # print(X_out[:5])
        # print("T_out")
        # print(T_out[:5])
        return X_out, T_out

    def train_level1(self):
        level = LEVEL_1
        # features = importance_features(model, X, Y, IMPORTANT_THRESHOLD)
        print("Training for non-scale data ...")
        X = self.train_set
        Y = self.target
        T = self.eval_set
        X_out_no_scale, T_out_no_scale = self.train_level1_single_dataset(
            X, Y, T)

        print("Training for scaled data ...")
        X = self.train_set_scale
        Y = self.target_log
        T = self.eval_set_scale
        X_out_scale, T_out_scale = self.train_level1_single_dataset(X, Y, T)

        print("Combine 2 sets of data")
        X_out = np.concatenate((X_out_no_scale, X_out_scale), axis=1)
        T_out = np.concatenate((T_out_no_scale, T_out_scale), axis=1)
        print(X_out[:5])
        self.save_train_data(level, X_out, Y, T_out)
        return X_out, T_out

    def train_level2(self, load_data=True):
        print("Training for level 2 ...")
        level = LEVEL_2
        if load_data:
            X, Y, T = self.load_pretrained_data(LEVEL_1)
        models_level2 = self.build_models_level2()
        # self.search_best_params(models_level2, X, Y)
        X_out, T_out = self.model_stack_train(models_level2, X, Y, T)
        print(X_out[:5])
        self.save_train_data(level, X_out, Y, T_out)
        return X_out, T_out

    # train and predict with given model, X: input, Y:label, T: test set
    def model_train_predict(self, model, X_in, Y_in, T_in):
        x_train, x_test, y_train, y_test = train_test_split(
            X_in, Y_in, test_size=0.31, random_state=324)
        print("Trainning ...")
        model.fit(x_train, y_train)
        y_test_pred = model.predict(x_test)
        print("rmse:", self.rmse(y_test, y_test_pred))
        y_pred = model.predict(T_in)[:]
        return y_pred

    def model_stacking(self, model_choice=1, boost=False, level=LEVEL_1, load_data=True):
        models = []
        if model_choice == 1:
            model = GradientBoostingRegressor(
                n_estimators=100, max_depth=1, loss='huber')
            model_dict = {
                'model': model,
                'model_param': {"n_estimators": [50, 100, 200, 500],
                                "max_depth": [1, 5, 10, 50],
                                "loss": ['ls', 'lad', 'huber', 'quantile']
                                },
                'boost': boost
            }
            models.append(model_dict.copy())
        elif model_choice == 2:
            print("Model stacking using GBM....")
            model = XGBRegressor(n_estimators=100, max_depth=1, n_jobs=-1)
            # model_name = model.__class__.__name__
            model_dict = {
                'model': model,
                'model_param': {"n_estimators": [50, 100, 200, 500],
                                "max_depth": [1, 5, 10, 50],
                                },
                'boost': boost
            }
            models.append(model_dict.copy())
        else:
            print("Model choice: (1 - GBM), (2-XGboost)")
            quit()

        model_name = model.__class__.__name__
        print("Model stacking using:", model_name, " boost:", boost)
        if load_data:
            X, Y, T = self.load_pretrained_data(level)
        # self.search_best_params(models, X, Y)
        if boost:
            model_temp = model
            model = AdaBoostRegressor(
                base_estimator=model_temp, n_estimators=500, learning_rate=0.1, random_state=200)
        y_prediction = self.model_train_predict(model, X, Y, T)
        self.export_prediction(y_prediction)

    def model_stacking_deepNN(self):
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

    # Kfold, Fit and predict for model
    def kfold_fit_predict(self, model, X, Y, T):
        kfolds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=321)
        total_rmse = 0
        for j, (train_idx, test_idx) in enumerate(kfolds.split(X)):
            X_train = X[train_idx]
            y_train = Y[train_idx]
            X_holdout = X[test_idx]
            y_holdout = Y[test_idx]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_holdout)[:]
            rmse1 = self.rmse(y_holdout, y_pred)
            total_rmse = total_rmse + rmse1
            print("fold:", j, "rmse:", rmse1)
        print("Avg rmse:", total_rmse / (j + 1))

        print("Predict final value")
        y_pred = model.predict(T)[:]
        return y_pred

    def model_kfold_xgboost(self):
        # rmse: 0.12772005156128946
        # Best score: 0.13676

        print("Model stacking using xgboosting....")
        model = XGBRegressor(n_estimators=500, max_depth=3,
                             n_jobs=-1, random_state=123)
        model_boost = AdaBoostRegressor(
            base_estimator=model, n_estimators=200, random_state=200)
        # model_boost.fit(x_train, y_train)
        X = self.train_set
        Y = self.target_log
        T = self.eval_set
        features = X.columns.values
        X_in = X[features].values
        Y_in = Y.values
        T_in = T[features].values
        y_prediction = self.kfold_fit_predict(model, X_in, Y_in, T_in)
        print(y_prediction[:5])
        self.export_prediction(y_prediction)

    def export_prediction(self, y_prediction):
        # Transform SalePrice to normal
        Y_eval = np.exp(y_prediction.ravel())
        print("Prediction:")
        print(Y_eval[:5])
        # save predicted sale price to CSV
        eval_output = pd.DataFrame(
            {'Id': self.eval_data_id, self.label: Y_eval})
        print("Evaluation output len:", len(eval_output))
        today = str(datetime.date.today())
        eval_output.to_csv(DATA_DIR + '/' + today + '-' +
                           'stack' + '.csv', index=False)


# ---- Main program --------------
stack_regression = StackRegression('SalePrice')
option = 4
if option == 1:
    # Run from begining to level 1
    stack_regression.load_data()
    stack_regression.transform_data()
    stack_regression.train_level1()
elif option == 2:
    # Train data for model level 2. Must run option = 1 first
    stack_regression.train_level2()
elif option == 3:
    # load data from level 1 and predict. Must run option = 1 first
    stack_regression.model_stacking(model_choice=1, level=LEVEL_1)
elif option == 4:
    # load data from level 2 and predict. Must run option = 2 first
    stack_regression.model_stacking(model_choice=1, level=LEVEL_2)
elif option == 5:
    stack_regression.load_data()
    stack_regression.transform_data()
    stack_regression.model_kfold_xgboost()
