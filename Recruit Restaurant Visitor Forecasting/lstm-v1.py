# Recruit Restaurant Visitor Forecasting
# https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting/
# https://www.kaggle.com/yekenot/explore-ts-with-lstm
# https://www.kaggle.com/meli19/py-single-light-gbm-lb-0-521/code
# https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
# https://machinelearningmastery.com/time-series-forecasting-supervised-learning/
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


# # Scikit-learn
from sklearn.model_selection import cross_val_score, train_test_split, learning_curve, validation_curve, KFold
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopword2s
from sklearn import decomposition

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
KERAS_N_ROUNDS = 50
KERAS_BATCH_SIZE = 64
KERAS_NODES = 64
KERAS_DROPOUT_RATE = 0.5  # => Best
# KERAS_DROPOUT_RATE = 0.2
# KERAS_REGULARIZER = KERAS_LEARNING_RATE/10
KERAS_REGULARIZER = 0.04
KERAS_VALIDATION_SPLIT = 0.2
KERAS_EARLY_STOPPING = 2
KERAS_MAXNORM = 3
KERAS_PREDICT_BATCH_SIZE = 4096
decay = KERAS_LEARNING_RATE / KERAS_N_ROUNDS
VERBOSE = True
model_weight_path = OUTPUT_DIR + "/model_weight.h5"

# frame a sequence as a supervised learning problem

# https://machinelearningmastery.com/time-series-forecasting-supervised-learning/


def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag + 1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


def scale(data):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = data.values
    scaler = scaler.fit(data)
    # transform train
    data = data.reshape(data.shape[0], data.shape[1])
    data_scaled = scaler.transform(data)
    return scaler, data_scaled

# inverse scaling for a forecasted value


def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


def keras_rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def model_lstm(X, batch_size):
    model = Sequential()
    model.add(LSTM(KERAS_NODES, batch_input_shape=(
        batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dropout(KERAS_DROPOUT_RATE, seed=12345))
    model.add(Dense(1))
    # compile the model
    optimizer = Adam(lr=KERAS_LEARNING_RATE, decay=decay)
    model.compile(optimizer=optimizer, loss=keras_rmse)
    print(model.summary())
    return model

# make a one-step forecast


def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]


def train_model(train, test):
    data = train[label]
    print("Shape Before:", data.shape)
    data = timeseries_to_supervised(data)
    print("Shape After:", data.shape)
    print(data[:5])
    scaler, data_scale = scale(data)
    print("After scaled:", data_scale[:5])
    X, y = data_scale[:, 0:-1], data_scale[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=KERAS_VALIDATION_SPLIT, shuffle=False)
    batch_size = 1
    model = model_lstm(X, batch_size)
    # Use Early-Stopping
    callback_early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=KERAS_EARLY_STOPPING, verbose=VERBOSE, mode='auto')
    callback_tensorboard = keras.callbacks.TensorBoard(
        log_dir=OUTPUT_DIR + '/tensorboard', histogram_freq=1, batch_size=KERAS_BATCH_SIZE,
        write_graph=True, write_grads=True, write_images=False)
    callback_checkpoint = keras.callbacks.ModelCheckpoint(
        model_weight_path, monitor='val_loss', verbose=VERBOSE, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(model)
    for i in range(KERAS_N_ROUNDS):
        logger.debug("Round:" + str(i + 1))
        history = model.fit(X_train, Y_train,
                            validation_data=(X_test, Y_test),
                            batch_size=batch_size,
                            epochs=1,
                            # callbacks=[
                            #     callback_early_stopping,
                            #     callback_checkpoint,
                            # ],
                            shuffle=False,
                            verbose=VERBOSE
                            )
        model.reset_states()
        history_val_loss = history.history['val_loss'][0]
        # if early_stopping.check(history_val_loss) is False:
        #     break

    # Load best model
    # model.load_weights(model_weight_path)
    # forecast the entire training dataset to build up state for forecasting
    train_reshaped = data_scale[:, 0].reshape(len(data_scale), 1, 1)
    Y_pred = model.predict(train_reshaped, batch_size=1)
    print(Y_pred.shape)
    print(Y_pred[-5:])
    X = np.array([Y_pred[-1]])
    # Predict
    predictions = list()
    for i in range(len(test)):
        # make one-step forecast
        yhat = forecast_lstm(model, 1, X)
        X1 = np.array([yhat])
        Y_pred = np.append(Y_pred, X1)
        # invert scaling
        yhat = invert_scale(scaler, X, yhat)
        # store forecast
        predictions.append(yhat)
        X = X1
    logger.debug("Test size:" + str(len(predictions)))    
    print(predictions[-5:])
    predictions = np.array(predictions).reshape(len(predictions), 1)
    print(Y_pred.shape)
    #  Plot y_pred
    plt.plot(y)
    plt.plot(Y_pred)
    plt.title('Forecasting')
    plt.ylabel('visitor')
    plt.xlabel('sequence')
    # plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(DATA_DIR + "/forecast.png")
    return predictions


class EarlyStopping():
    def __init__(self, model):
        self.val_loss = math.inf
        self.model = model

    def check(self, val_loss):
        if val_loss <= self.val_loss:
            logger.debug("Val loss improved from:" +
                         str(self.val_loss) + " to:" + str(val_loss))
            self.val_loss = val_loss
            self.model.save_weights(model_weight_path)
            return True
        else:
            logger.debug("Val loss not imporved. End trainning")
            return False


# ---------------- Main -------------------------
if __name__ == "__main__":
    start = time.time()
    pd.options.display.float_format = '{:,.5f}'.format
    logger = logging.getLogger('mercari-price')
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

    # Load train and test data
    train = pd.read_csv(DATA_DIR + '/train.csv')
    test = pd.read_csv(DATA_DIR + '/test.csv')
    logger.debug("Data set columns")
    logger.debug(train.columns)
    logger.debug(test.columns)
    store_id_list = np.unique(test['air_store_id'])
    logger.debug("Number of stores:" + str(len(store_id_list)))
    cols = ['visit_date', 'air_store_id', 'visitors']
    label = 'visitors'
    test_cols = ['id'] + cols
    train_data = train[cols].copy()
    test_data = test[test_cols].copy()
    count = 0
    for air_store_id in store_id_list:
        logger.debug("Forecasting for:" + str(air_store_id))
        count += 1
        train_visit_date = train_data[train_data.air_store_id ==
                                      air_store_id].visit_date
        test_visit_date = test_data[test_data.air_store_id ==
                                    air_store_id].visit_date
        logger.debug("visit_date:" + str(np.min(train_visit_date)) + ":" + str(np.max(train_visit_date)
                                                                               ) + ":" + str(np.min(test_visit_date)) + ":" + str(np.max(test_visit_date)))
        train_set = train_data[train_data.air_store_id == air_store_id]
        test_set = test_data[test_data.air_store_id == air_store_id]
        print("Test size:", len(test_set))
        y_pred = train_model(train_set, test_set)
        print("Y pred size:", y_pred.shape)
        test_data.loc[test_data.air_store_id == air_store_id, label] = y_pred
        # test_set[label] = y_pred
        if count > 0:
            break
    end = time.time() - start
    logger.info("Total time:" + str(end))
    logger.info("Done!")
