# Recruit Restaurant Visitor Forecasting
# https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting/
# https://www.kaggle.com/yekenot/explore-ts-with-lstm
# https://www.kaggle.com/meli19/py-single-light-gbm-lb-0-521/code
# https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
# https://machinelearningmastery.com/time-series-forecasting-supervised-learning/
# https://github.com/gcarq/keras-timeseries-prediction/blob/master/main.py
# System
import datetime as dtime
import time
import logging
import sys
import os
import pickle
import glob
import psutil
from multiprocessing import Pool, Manager, SimpleQueue
import math
from tqdm import trange
# set this code for reproduce training result
os.environ['PYTHONHASHSEED'] = '0'

# data processing
# use matplotlib without X server
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
import numpy as np
import random as rn

# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.

np.random.seed(42)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.

rn.seed(12345)

# ML
# Tensorflow
import tensorflow as tf
# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/set_random_seed
tf.set_random_seed(1234)
# Set GPU memory


def set_gpu_memory(gpu_fraction=0.1):
    print("Set GPU memory with faction:" + str(gpu_fraction))
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


session = set_gpu_memory()

# # Scikit-learn
from sklearn.model_selection import cross_val_score, train_test_split, learning_curve, validation_curve, KFold
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopword2s
from sklearn import decomposition
from sklearn.metrics import mean_squared_error

# Keras
import keras
from keras.models import Sequential, Model
from keras.layers import concatenate, Input, Dense, Dropout, BatchNormalization, Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D, LSTM, CuDNNLSTM, Bidirectional, GlobalAveragePooling1D, Reshape, Dot, Average, TimeDistributed, GRU, Activation, CuDNNGRU
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from keras.layers.embeddings import Embedding
import keras.backend.tensorflow_backend as KTF
from keras import backend as K
from keras.utils import plot_model
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints
from keras.models import load_model

# Constants
# Input data files are available in the DATA_DIR directory.
DATA_DIR = "data-temp"
OUTPUT_DIR = DATA_DIR
KERAS_LEARNING_RATE = 0.001
KERAS_N_ROUNDS = 100
KERAS_NODES = 512
# KERAS_DROPOUT_RATE = 0.5  # => Best
KERAS_DROPOUT_RATE = 0.2
# KERAS_REGULARIZER = KERAS_LEARNING_RATE/10
KERAS_REGULARIZER = 0.04
KERAS_EARLY_STOPPING = 5
# KERAS_METRIC_DELTA = 0.01
KERAS_METRIC_DELTA = 0.003
KERAS_MAXNORM = 3
decay = KERAS_LEARNING_RATE / KERAS_N_ROUNDS
VERBOSE = 2
model_weight_path = OUTPUT_DIR + "/model_weight.h5"


# frame a sequence as a supervised learning problem


def create_dataset(dataset: np.ndarray, look_back: int=1) -> (np.ndarray, np.ndarray):
    # https://github.com/gcarq/keras-timeseries-prediction/blob/master/main.py
    """
    The function takes two arguments: the `dataset`, which is a NumPy array that we want to convert into a dataset,
    and the `look_back`, which is the number of previous time steps to use as input variables
    to predict the next time period — in this case defaulted to 1.
    :param dataset: numpy dataset
    :param look_back: number of previous time steps as int
    :return: tuple of input and output dataset
    """
    data_x, data_y = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        data_x.append(a)
        data_y.append(dataset[i + look_back, 0])
    return np.array(data_x), np.array(data_y)


def split_dataset(dataset: np.ndarray, train_size, look_back) -> (np.ndarray, np.ndarray):
    # https://github.com/gcarq/keras-timeseries-prediction/blob/master/main.py
    """
    Splits dataset into training and test datasets. The last `look_back` rows in train dataset
    will be used as `look_back` for the test dataset.
    :param dataset: source dataset
    :param train_size: specifies the train data size
    :param look_back: number of previous time steps as int
    :return: tuple of training data and test dataset
    """
    if not train_size > look_back:
        raise ValueError('train_size must be lager than look_back')
    train, test = dataset[0:train_size,
                          :], dataset[train_size - look_back:len(dataset), :]
    # print('train_dataset: {}, test_dataset: {}'.format(len(train), len(test)))
    return train, test


def plot_data(dataset: np.ndarray,
              look_back: int,
              train_predict: np.ndarray,
              test_predict: np.ndarray,
              forecast_predict: np.ndarray):
    # https://github.com/gcarq/keras-timeseries-prediction/blob/master/main.py
    """
    Plots baseline and predictions.
    blue: baseline
    green: prediction with training data
    red: prediction with test data
    cyan: prediction based on predictions
    :param dataset: dataset used for predictions
    :param look_back: number of previous time steps as int
    :param train_predict: predicted values based on training data
    :param test_predict: predicted values based on test data
    :param forecast_predict: predicted values based on previous predictions
    :return: None
    """
    plt.plot(dataset)
    plt.plot([None for _ in range(look_back)] +
             [x for x in train_predict])
    plt.plot([None for _ in range(look_back)] +
             [None for _ in train_predict] +
             [x for x in test_predict])
    plt.plot([None for _ in range(look_back)] +
             [None for _ in train_predict] +
             [None for _ in test_predict] +
             [x for x in forecast_predict])
    plt.savefig(DATA_DIR + "/forecast.png")
    plt.clf()


def plot_history(history):
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model rmse')
    plt.ylabel('rmse')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(DATA_DIR + "/history.png")
    plt.clf()


def rmsle(y, pred):
    # return mean_squared_error(np.log(y + 1), np.log(pred + 1))**0.5
    return np.sqrt(np.square(np.log(y + 1) - np.log(pred + 1)).mean())


def keras_rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def model_lstm(look_back: int, batch_size: int=1):
    model = Sequential()
    # model.add(LSTM(KERAS_NODES, activation='relu', batch_input_shape=(
    #     batch_size, look_back, 1), stateful=True, return_sequences=False))
    model.add(CuDNNLSTM(KERAS_NODES, batch_input_shape=(
        batch_size, look_back, 1), stateful=True, return_sequences=False))
    model.add(Dropout(KERAS_DROPOUT_RATE, seed=12345))
    model.add(Dense(1))
    # compile the model
    optimizer = Adam(lr=KERAS_LEARNING_RATE)
    model.compile(optimizer=optimizer, loss=keras_rmse)
    print(model.summary())
    return model


def make_forecast(model: Sequential, look_back_buffer: np.ndarray, timesteps: int=1, batch_size: int=1):
    # https://github.com/gcarq/keras-timeseries-prediction/blob/master/main.py
    forecast_predict = np.empty((0, 1), dtype=np.float32)
    for _ in trange(timesteps, desc='predicting data\t', mininterval=1.0):
        # make prediction with current lookback buffer
        cur_predict = model.predict(look_back_buffer, batch_size)
        # add prediction to result
        forecast_predict = np.concatenate(
            [forecast_predict, cur_predict], axis=0)
        # add new axis to prediction to make it suitable as input
        cur_predict = np.reshape(
            cur_predict, (cur_predict.shape[1], cur_predict.shape[0], 1))
        # remove oldest prediction from buffer
        look_back_buffer = np.delete(look_back_buffer, 0, axis=1)
        # concat buffer with newest prediction
        look_back_buffer = np.concatenate(
            [look_back_buffer, cur_predict], axis=1)
    return forecast_predict


def train_model(train_set, test_set):
    # https://github.com/gcarq/keras-timeseries-prediction/blob/master/main.py
    start = time.time()
    logger.debug("Tranform train/test data ...")
    data = train_set.values
    data = data.reshape(data.shape[0], 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(data)
    # split into train and test sets
    # look_back = int(len(dataset) * 0.20)
    train_size = int(len(dataset) * 0.70)
    train, test = split_dataset(dataset, train_size, look_back)
    logger.debug("Train size:" + str(len(train)) + ". Test size:" +
                 str(len(test)) + ". Look back:" + str(look_back))
    # reshape into X=t and Y=t+1
    train_x, train_y = create_dataset(train, look_back)
    test_x, test_y = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
    # create and fit Multilayer Perceptron model
    logger.debug("Trainning ....")
    early_stopping = EarlyStopping(
        model, KERAS_METRIC_DELTA, KERAS_EARLY_STOPPING)
    history_all = {'loss': [], 'val_loss': []}
    # Reset weight before training
    model.set_weights(model_init_weights)
    for i in range(KERAS_N_ROUNDS):
        logger.debug("Round:" + str(i + 1) + "/" + str(KERAS_N_ROUNDS))
        history = model.fit(train_x, train_y, validation_data=(
            test_x, test_y), epochs=1, batch_size=batch_size, verbose=VERBOSE, shuffle=False)
        history_all['loss'].append(history.history['loss'])
        history_all['val_loss'].append(history.history['val_loss'])
        model.reset_states()
        if early_stopping.check(i + 1, history.history['val_loss'][0]) is False:
            break

    plot_history(history_all)
    # Load best model
    model.load_weights(model_weight_path)
    # generate predictions for training
    train_predict = model.predict(train_x, batch_size)
    test_predict = model.predict(test_x, batch_size)
    # generate forecast predictions
    logger.debug("Forecasting for:" + str(len(test_set)))
    forecast_predict = make_forecast(
        model, test_x[-1::], timesteps=len(test_set), batch_size=batch_size)
    # invert dataset and predictions
    dataset = scaler.inverse_transform(dataset)
    train_predict = scaler.inverse_transform(train_predict)
    train_y = scaler.inverse_transform([train_y])
    test_predict = scaler.inverse_transform(test_predict)
    test_y = scaler.inverse_transform([test_y])
    forecast_predict = scaler.inverse_transform(forecast_predict)
    print(forecast_predict[-5:])
    plot_data(dataset, look_back, train_predict,
              test_predict, forecast_predict)
    # calculate root mean squared error
    train_score = rmsle(train_y[0], train_predict[:, 0])
    test_score = rmsle(test_y[0], test_predict[:, 0])
    logger.debug('Train/Test RMSLE: ' +
                 str(train_score) + " - " + str(test_score))
    end = time.time() - start
    logger.info("Training time:" + str(end))
    return forecast_predict


class EarlyStopping():
    def __init__(self, model, delta=0, patience=0, verbose=False):
        self.metric = math.inf
        self.model = model
        self.delta = delta
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.best_epoch = 0

    def check(self, epoch, metric):
        if not math.isinf(metric):
            metric_delta = metric - self.delta
            if metric_delta <= self.metric:
                if self.verbose:
                    logger.debug("metric improved from:" +
                                 str(self.metric) + " to:" + str(metric) + " (-" + str(self.delta) + ")")
                self.metric = metric
                self.wait = 0
                self.best_epoch = epoch
                self.model.save_weights(model_weight_path)
            else:
                if self.patience > 0:
                    self.wait += 1
                    if self.wait >= self.patience:
                        logger.debug("metric not improved for:" +
                                     str(self.patience) + ". End trainning")
                        logger.debug("Best epoch:" + str(self.best_epoch))
                        return False
        else:
            logger.debug("Metric is Inf. End training")
            return False
        return True


# ---------------- Main -------------------------
if __name__ == "__main__":
    start = time.time()
    pd.options.display.float_format = '{:,.5f}'.format
    logger = logging.getLogger('recruit-forecasting')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(OUTPUT_DIR + '/model.log', mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # set GPU memory
    KTF.set_session(session)

    # Load train and test data
    train = pd.read_csv(DATA_DIR + '/train_2.csv')
    test = pd.read_csv(DATA_DIR + '/test_2.csv')
    logger.debug("Data set columns")
    logger.debug(train.columns)
    logger.debug(test.columns)
    store_id_list = np.unique(test['air_store_id'])
    logger.debug("Number of stores:" + str(len(store_id_list)))
    cols = ['visit_date', 'air_store_id', 'visitors']
    label = 'visitors'
    test_cols = ['id'] + cols
    count = 0
    batch_size = 1
    look_back = 10

    # Build model
    model = model_lstm(look_back, batch_size=batch_size)
    # Save initilize model weigth for reseting weigth after each loop
    model_init_weights = model.get_weights()

    for air_store_id in store_id_list:
        count += 1
        logger.debug("Processing for store:" + str(air_store_id) +
                     ". Order:" + str(count) + "/" + str(len(store_id_list)))
        train_set = train[train.air_store_id == air_store_id][label]
        test_set = test[test.air_store_id == air_store_id][label]
        try:
            y_pred = train_model(train_set, test_set)
            test.loc[test.air_store_id == air_store_id, label] = y_pred
        except:
            logger.debug("Error training for store:" + str(air_store_id) + ". Try one more time")
            # Try to train 1 more time
            try:
                y_pred = train_model(train_set, test_set)
                test.loc[test.air_store_id == air_store_id, label] = y_pred
            except:
                logger.debug("Error training for store:" + str(air_store_id))

    # Debug
    # air_store_id = 'air_3155ee23d92202da'
    # logger.debug("Processing for store:" + str(air_store_id) +
    #              ". Order:" + str(count) + "/" + str(len(store_id_list)))
    # train_set = train[train.air_store_id == air_store_id][label]
    # test_set = test[test.air_store_id == air_store_id][label]
    # y_pred = train_model(train_set, test_set)
    # test.loc[test.air_store_id == air_store_id, label] = y_pred

    # convert negative values to abs values
    test.loc[test[label] < 0, label] = np.absolute(
        test[test[label] < 0][label])
    logger.debug("Save submission")
    eval_output = test[['id', label]]
    today = str(dtime.date.today())
    logger.debug("Date:" + today)
    eval_output.to_csv(
        OUTPUT_DIR + '/' + today + '-submission.csv', index=False, float_format='%.7f')
    end = time.time() - start
    logger.info("Total time:" + str(end))
    logger.info("Done!")
