# Recruit Restaurant Visitor Forecasting
# https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting/
# https://www.kaggle.com/yekenot/explore-ts-with-lstm
# https://www.kaggle.com/meli19/py-single-light-gbm-lb-0-521/code
# https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/
# https://machinelearningmastery.com/time-series-forecasting-supervised-learning/
# https://github.com/gcarq/keras-timeseries-prediction/blob/master/main.py
# System
import multiprocessing
if __name__ == "__main__":
    # Using spawn method to prevent TensorFlow hangup when start new process
    # https://github.com/tensorflow/tensorflow/issues/5448
    multiprocessing.set_start_method('spawn')
from tqdm import tqdm
tqdm.monitor_interval = 0  # Prevent error in tqdm

import datetime as dtime
import time
import logging
import sys
import os
import pickle
import glob
import psutil
import math
from tqdm import trange
from importlib import import_module
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

# # Scikit-learn
from sklearn.model_selection import cross_val_score, train_test_split, learning_curve, validation_curve, KFold
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopword2s
from sklearn import decomposition
from sklearn.metrics import mean_squared_error

# ML
# Tensorflow
import tensorflow as tf
# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
# For further details, see:
# https://www.tensorflow.org/api_docs/python/tf/set_random_seed
tf.set_random_seed(1234)
# Set GPU memory


def set_gpu_memory(gpu_fraction=0.05):
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

KTF.set_session(session)   # Set GPU Memory

# Constants
# Input data files are available in the DATA_DIR directory.
LOG_LEVEL = logging.DEBUG
DATA_DIR = "data-temp"
OUTPUT_DIR = DATA_DIR
NUM_THREAD = 4
KERAS_LEARNING_RATE = 0.001
KERAS_N_ROUNDS = 100
KERAS_NODES = 512
# KERAS_NODES = 256
KERAS_BATCH_SIZE = 1
KERAS_LOOK_BACK = 7
# KERAS_DROPOUT_RATE = 0.5  # => Best
KERAS_DROPOUT_RATE = 0.2
# KERAS_REGULARIZER = KERAS_LEARNING_RATE/10
KERAS_REGULARIZER = 0.04
KERAS_EARLY_STOPPING = 5
# KERAS_METRIC_DELTA = 0.01
KERAS_METRIC_DELTA = 0.002
KERAS_MAXNORM = 3
KERAS_RETRY = 3  # number of fitting retries to find best score
decay = KERAS_LEARNING_RATE / KERAS_N_ROUNDS
VERBOSE = 1


def rmsle(y, pred):
    # return mean_squared_error(np.log(y + 1), np.log(pred + 1))**0.5
    return np.sqrt(np.square(np.log(y + 1) - np.log(pred + 1)).mean())


class EarlyStopping():
    def __init__(self, logger, model, model_weight_path, delta=0, patience=0, verbose=False):
        self.metric = math.inf
        self.logger = logger
        self.model = model
        self.model_weight_path = model_weight_path
        self.delta = delta
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.best_epoch = 0
        self.model_weight = self.model.get_weights()

    def check(self, epoch, metric):
        if not math.isinf(metric):
            metric_delta = metric - self.delta
            if metric_delta <= self.metric:
                if self.verbose:
                    self.logger.debug("metric improved from:" +
                                      str(self.metric) + " to:" + str(metric) + " (-" + str(self.delta) + ")")
                self.metric = metric
                self.wait = 0
                self.best_epoch = epoch
                # self.model.save_weights(self.model_weight_path)
                self.model_weight = self.model.get_weights()
            else:
                if self.patience > 0:
                    self.wait += 1
                    if self.wait >= self.patience:
                        self.logger.debug("metric not improved for:" +
                                          str(self.patience) + ". End training")
                        self.logger.debug("Best epoch:" + str(self.best_epoch))
                        return False
        else:
            self.logger.error("Metric is Inf. End training")
            return False
        return True


class MultiProcessBase(multiprocessing.Process):
    def __init__(self, thread_count=0, in_queue=None, out_queue=None):
        multiprocessing.Process.__init__(self)
        self.thread_count = thread_count
        if in_queue is not None:
            self.in_queue = in_queue
        if out_queue is not None:
            self.out_queue = out_queue

    def get_logger(self):
        logger = logging.getLogger('kaggle-' + str(self.thread_count))
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - ' + str(self.thread_count) + ': %(message)s')
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(LOG_LEVEL)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        # create file handler which logs even debug messages
        fh = logging.FileHandler(
            DATA_DIR + '/model_' + str(self.thread_count) + '.log', mode='a')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger


class TimeSeriesForecastMP(MultiProcessBase):
    def __init__(self, thread_count=0, in_queue=None, out_queue=None):
        MultiProcessBase.__init__(self, thread_count, in_queue, out_queue)
        self.model_weight_path = OUTPUT_DIR + \
            "/model_weight" + str(self.thread_count) + ".h5"

    def run(self):
        self.logger = self.get_logger()
        logger = self.logger
        # Build model
        model = self.model_lstm(look_back=KERAS_LOOK_BACK,
                                batch_size=KERAS_BATCH_SIZE)
        # # Save initilize model weigth for reseting weigth after each loop
        self.model_init_weights = model.get_weights()
        while True:
            # Get the work from the queue and expand the tuple
            data = self.in_queue.get()
            if data is None:
                logger.info(str(self.thread_count) +
                            " No task remain. Existing ...")
                self.in_queue.task_done()
                break
            logger.debug(
                "Store-ID: %d/%d - %s", data['store_order'], data['store_count'],
                data['air_store_id'])
            # self.train_model(model, data)
            try:
                self.train_model(model, data)
            except Exception as e:
                logger.error("Error 1 - %s: %s", data['air_store_id'], e)
            #     logger.debug("Try one more time")
            #     # Try to train 1 more time
            #     try:
            #         self.train_model(model, data)
            #     except Exception as e:
            #         logger.error("Error 2 - %s: %s", data['air_store_id'], e)

            # Reset weight after training
            model.set_weights(self.model_init_weights)
            self.in_queue.task_done()

    def set_gpu_memory(self, gpu_fraction=0.1):
        print("Set GPU memory with faction:" + str(gpu_fraction))
        '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

        num_threads = os.environ.get('OMP_NUM_THREADS')
        gpu_options = self.tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_fraction, allow_growth=True)

        if num_threads:
            return self.tf.Session(config=self.tf.ConfigProto(
                gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
        else:
            return self.tf.Session(config=self.tf.ConfigProto(gpu_options=gpu_options))

    # frame a sequence as a supervised learning problem
    def create_dataset(self, dataset: np.ndarray, look_back: int=1) -> (np.ndarray, np.ndarray):
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

    def split_dataset(self, dataset: np.ndarray, train_size, look_back) -> (np.ndarray, np.ndarray):
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

    def plot_data(self, dataset: np.ndarray,
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
        plt.savefig(DATA_DIR + "/forecast_" + str(self.thread_count) + ".png")
        plt.clf()

    def plot_history(self, history):
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model rmse')
        plt.ylabel('rmse')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(DATA_DIR + "/history_" + str(self.thread_count) + ".png")
        plt.clf()

    def keras_rmse(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

    def model_lstm(self, look_back: int, batch_size: int=1):
        model = Sequential()
        # model.add(LSTM(KERAS_NODES, activation='relu', batch_input_shape=(
        # batch_size, look_back, 1), stateful=True, return_sequences=False))
        model.add(CuDNNLSTM(KERAS_NODES, batch_input_shape=(
            batch_size, look_back, 1), stateful=True, return_sequences=False))
        model.add(Dropout(KERAS_DROPOUT_RATE, seed=12345))
        model.add(Dense(1))
        # compile the model
        optimizer = Adam(lr=KERAS_LEARNING_RATE)
        model.compile(optimizer=optimizer, loss=self.keras_rmse)
        print(model.summary())
        return model

    def make_forecast(self, model, look_back_buffer, timesteps=1, batch_size=1):
        # https://github.com/gcarq/keras-timeseries-prediction/blob/master/main.py
        logger = self.logger
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

    def train_model(self, model, in_data):
        # https://github.com/gcarq/keras-timeseries-prediction/blob/master/main.py
        start = time.time()
        logger = self.logger
        look_back = KERAS_LOOK_BACK
        batch_size = KERAS_BATCH_SIZE
        air_store_id, store_order, store_count, train_set, test_set = in_data['air_store_id'], in_data[
            'store_order'], in_data['store_count'], in_data['train_set'], in_data['test_set']
        logger.debug("Tranform train/test data ...")
        data = train_set.values
        data = data.reshape(data.shape[0], 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset = scaler.fit_transform(data)
        # split into train and test sets
        # look_back = int(len(dataset) * 0.20)
        train_size = int(len(dataset) * 0.70)
        # train_size = len(dataset) - look_back
        train, test = self.split_dataset(dataset, train_size, look_back)
        logger.debug("Train size:" + str(len(train)) + ". Test size:" +
                     str(len(test)) + ". Look back:" + str(look_back))
        # reshape into X=t and Y=t+1
        train_x, train_y = self.create_dataset(train, look_back)
        test_x, test_y = self.create_dataset(test, look_back)
        # reshape input to be [samples, time steps, features]
        train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))
        test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))
        # create and fit Multilayer Perceptron model
        logger.debug("Training for " + str(air_store_id) + " ....")
        # model = self.model_lstm(look_back, batch_size=KERAS_BATCH_SIZE)
        history_all = {'loss': [], 'val_loss': []}
        model_weights = []
        metrics = []
        desc = str(self.thread_count) + " - " + str(store_order) + \
            "/" + str(store_count) + ':Fitting\t'
        for retry in range(KERAS_RETRY):
            logger.debug("Retry %d", retry + 1)
            early_stopping = EarlyStopping(logger,
                                       model, self.model_weight_path,
                                       KERAS_METRIC_DELTA,
                                       KERAS_EARLY_STOPPING)    
            for i in trange(KERAS_N_ROUNDS, desc=desc, mininterval=1.0):
                if VERBOSE > 0:
                    logger.debug("Round:" + str(i + 1) + "/" + str(KERAS_N_ROUNDS))
                history = model.fit(train_x, train_y, validation_data=(
                    test_x, test_y), epochs=1, batch_size=batch_size, verbose=VERBOSE, shuffle=False)
                history_all['loss'].append(history.history['loss'])
                history_all['val_loss'].append(history.history['val_loss'])
                model.reset_states()
                if early_stopping.check(i + 1, history.history['val_loss'][0]) is False:
                    break
            logger.debug("End training for retry %s of store:%s. Best epoch:%d. Best metric:%f", retry + 1,
                        air_store_id, early_stopping.best_epoch, early_stopping.metric)
            metrics.append(early_stopping.metric)
            model_weights.append(early_stopping.model_weight)
            del early_stopping
            # Reset model weight
            model.set_weights(self.model_init_weights)

        
        # Get best retry    
        best_metric = np.min(metrics)
        best_retry = np.argmin(metrics)
        logger.debug("Best retry:%d. Best metric:%f",best_retry + 1, best_metric)

        # Load best model
        # model.load_weights(self.model_weight_path)
        model.set_weights(model_weights[best_retry])
        # generate predictions for training
        train_predict = model.predict(train_x, batch_size)
        test_predict = model.predict(test_x, batch_size)
        # generate forecast predictions
        logger.debug("Forecasting for:" + str(len(test_set)))
        forecast_predict = self.make_forecast(
            model, test_x[-1::], timesteps=len(test_set), batch_size=batch_size)
        # invert dataset and predictions
        forecast_predict = scaler.inverse_transform(forecast_predict)

        if VERBOSE > 0:
            dataset = scaler.inverse_transform(dataset)
            train_predict = scaler.inverse_transform(train_predict)
            train_y = scaler.inverse_transform([train_y])
            test_predict = scaler.inverse_transform(test_predict)
            test_y = scaler.inverse_transform([test_y])
            print(forecast_predict[-2:])
            self.plot_history(history_all)
            self.plot_data(dataset, look_back, train_predict,
                           test_predict, forecast_predict)
            # calculate root mean squared error
            train_score = rmsle(train_y[0], train_predict[:, 0])
            test_score = rmsle(test_y[0], test_predict[:, 0])
            logger.debug('Train/Test RMSLE: ' +
                         str(train_score) + " - " + str(test_score))
        out_data = {}
        out_data[air_store_id] = forecast_predict
        self.out_queue.put(out_data)
        # del model
        end = time.time() - start
        logger.info("Training time:" + str(end))


class Main(MultiProcessBase):
    def __init__(self, thread_count=0, in_queue=None, out_queue=None):
        MultiProcessBase.__init__(self, thread_count, in_queue, out_queue)

    def run(self):
        self.logger = self.get_logger()
        logger = self.logger
        logger.debug("Main process running...")
        test = pd.read_csv(DATA_DIR + '/test_2.csv')
        label = 'visitors'
        # Process output result
        count = 0
        while True:
            data = self.out_queue.get()
            if data is not None:
                for air_store_id in data:
                    count += 1
                    logger.debug(
                        "%d:Update forecasting result for: %s ", count, air_store_id)
                    y_pred = data[air_store_id]
                    test.loc[test.air_store_id == air_store_id, label] = y_pred
            else:
                break
        logger.debug("Total results: %d", count)
        # convert negative values to abs values
        test.loc[test[label] < 0, label] = np.absolute(
            test[test[label] < 0][label])
        logger.debug("Save submission")
        eval_output = test[['id', label]]
        today = str(dtime.date.today())
        logger.debug("Date:" + today)
        eval_output.to_csv(
            OUTPUT_DIR + '/' + today + '-submission.csv', index=False)


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
    # KTF.set_session(session)

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

    # Create global queue
    in_queue = multiprocessing.JoinableQueue()
    out_queue = multiprocessing.Queue()

    #  Init worker thread
    #  create worker
    workers = [TimeSeriesForecastMP(x + 1, in_queue=in_queue, out_queue=out_queue)
               for x in range(NUM_THREAD)]
    logger.info("Total workers:" + str(len(workers)))
    for w in workers:
        w.start()

    main_proc = Main(NUM_THREAD + 1, in_queue=in_queue, out_queue=out_queue)
    main_proc.start()

    store_id_list = ['air_900d755ebd2f7bbd']
    for air_store_id in store_id_list:
        count += 1
        logger.debug("Processing for store:" + str(air_store_id) +
                     ". Order:" + str(count) + "/" + str(len(store_id_list)))
        in_data = {}
        in_data['air_store_id'] = air_store_id
        in_data['store_order'] = count
        in_data['store_count'] = len(store_id_list)
        in_data['train_set'] = train[train.air_store_id == air_store_id][label]
        in_data['test_set'] = test[test.air_store_id == air_store_id][label]
        in_queue.put(in_data)
        # if count > 1:
        #     break

    # Debug
    # air_store_id = 'air_3155ee23d92202da'
    # logger.debug("Processing for store:" + str(air_store_id) +
    #              ". Order:" + str(count) + "/" + str(len(store_id_list)))
    # train_set = train[train.air_store_id == air_store_id][label]
    # test_set = test[test.air_store_id == air_store_id][label]
    # y_pred = train_model(train_set, test_set)
    # test.loc[test.air_store_id == air_store_id, label] = y_pred

    # Add a poison pill for each consumer
    for i in range(NUM_THREAD):
        in_queue.put(None)

    # Causes the main thread to wait for the queue to finish processing all
    # the tasks
    in_queue.join()
    # Signal end queue
    out_queue.put(None)

    #  Waiting some seconds for Main finishing
    time.sleep(10)
    logger.debug("Done all training and forecasting")

    end = time.time() - start
    logger.info("Total time:" + str(end))
    logger.info("Done!")
