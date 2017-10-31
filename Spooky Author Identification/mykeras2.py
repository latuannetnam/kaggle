# Spooky Author Identification
# Share code and discuss insights to identify horror authors from their writings
# https://www.kaggle.com/c/spooky-author-identification
# Credit:
# https://www.kaggle.com/knowledgegrappler/magic-embeddings-keras-a-toy-example
# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
# System
import datetime as dtime
import time
import logging
import sys
import os

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

import nltk.stem as stm
import re

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
# For further details, see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
tf.set_random_seed(1234)


# # Scikit-learn
from sklearn.model_selection import cross_val_score, train_test_split, learning_curve, validation_curve, KFold
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelBinarizer, StandardScaler


# Keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Flatten
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot
from keras.layers.embeddings import Embedding
import keras.backend.tensorflow_backend as KTF

# Constants
# Input data files are available in the DATA_DIR directory.
DATA_DIR = "data-temp"
GLOBAL_DATA_DIR = "/home/latuan/Programming/machine-learning/data"

VOCAB_SIZE = 50
SEQUENCE_LENGTH = 50
OUTPUT_DIM = 64
KERAS_LEARNING_RATE = 0.001
KERAS_N_ROUNDS = 1000
KERAS_BATCH_SIZE = 64
KERAS_NODES = 1024
KERAS_LAYERS = 1
KERAS_DROPOUT_RATE = 0.5
# KERAS_REGULARIZER = KERAS_LEARNING_RATE/10
KERAS_REGULARIZER = 0
KERAS_VALIDATION_SPLIT = 0.2
KERAS_EARLY_STOPPING = 10
KERAS_MAXNORM = 3
KERAS_PREDICT_BATCH_SIZE = 1024
VERBOSE = True
model_weight_path = DATA_DIR + "/model_weight.h5"
model_path = DATA_DIR + "/model.json"
random_state = 12343
decay = KERAS_LEARNING_RATE / KERAS_N_ROUNDS
nodes = KERAS_NODES
dropout = KERAS_DROPOUT_RATE

# disable GPU: export CUDA_VISIBLE_DEVICES=

pd.options.display.float_format = '{:,.4f}'.format


logger = logging.getLogger('safe-drive-prediction')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)
# create file handler which logs even debug messages
fh = logging.FileHandler(DATA_DIR + '/model.log', mode='a')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)


# Set GPU memory


def set_gpu_memory(gpu_fraction=0.3):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# set GPU memory
# KTF.set_session(set_gpu_memory())
# custom objective function (similar to auc)



# ---------------- Main -------------------------
if __name__ == "__main__":
    logger.info("Running ..")
    # Load data. Download
    # from:https://www.kaggle.com/c/nyc-taxi-trip-duration/data
    logger.info("Loading data ...")
    train_data = pd.read_csv(DATA_DIR + "/train.csv")
    eval_data = pd.read_csv(DATA_DIR + "/test.csv")

    logger.debug("train size:" + str(train_data.shape) +
                 " test size:" + str(eval_data.shape))
    label = 'author'
    eval_id = eval_data['id']

    # CREATE TARGET VARIABLE
    logger.debug("One hot encoding for label")
    train_data["EAP"] = (train_data.author == "EAP") * 1
    train_data["HPL"] = (train_data.author == "HPL") * 1
    train_data["MWS"] = (train_data.author == "MWS") * 1
    target_vars = ["EAP", "HPL", "MWS"]
    Y_train = train_data[target_vars].values

    logger.debug("Encoding text")
    # integer encode the documents
    train_data["encoded_text"] = train_data.text.apply(
        lambda x: one_hot(x, VOCAB_SIZE))
    eval_data["encoded_text"] = eval_data.text.apply(
        lambda x: one_hot(x, VOCAB_SIZE))
    print(train_data["encoded_text"][:3])

    # pad documents to a max length of 4 words

    X_train = pad_sequences(
        train_data["encoded_text"], maxlen=SEQUENCE_LENGTH, padding='post')
    X_eval = pad_sequences(
        eval_data["encoded_text"], maxlen=SEQUENCE_LENGTH, padding='post')
    print(X_train[:3])

    # define the model
    logger.debug("Model definition")
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, OUTPUT_DIM, input_length=SEQUENCE_LENGTH))
    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))
    # compile the model
    optimizer = Adam(lr=KERAS_LEARNING_RATE, decay=decay)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy', metrics=['acc'])
    # summarize the model
    print(model.summary())
    # fit the model
    # Use Early-Stopping
    callback_early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=KERAS_EARLY_STOPPING, verbose=VERBOSE, mode='auto')
    callback_tensorboard = keras.callbacks.TensorBoard(
        log_dir=DATA_DIR + '/tensorboard', histogram_freq=1, batch_size=KERAS_BATCH_SIZE,
        write_graph=True, write_grads=True, write_images=False)
    callback_checkpoint = keras.callbacks.ModelCheckpoint(
        model_weight_path, monitor='val_loss', verbose=VERBOSE, save_best_only=True, mode='auto')
    logger.info("Training ...")
    start = time.time()
    # Training model
    history = model.fit(X_train, Y_train,
                        validation_split=KERAS_VALIDATION_SPLIT,
                        batch_size=KERAS_BATCH_SIZE,
                        epochs=KERAS_N_ROUNDS,
                        callbacks=[
                            # callback_tensorboard,
                            callback_early_stopping,
                            callback_checkpoint,
                        ],
                        verbose=VERBOSE
                        )
    end = time.time() - start
    logger.debug("Train time:" + str(end))
    # load best model
    logger.info("Loading best model ...")
    model.load_weights(model_weight_path)
    logger.debug('Best metric:' + str(callback_early_stopping.best))
    logger.debug(
        'Best round:' + str(callback_early_stopping.stopped_epoch - KERAS_EARLY_STOPPING))
    # PREDICTION
    logger.debug("Prediction")
    Y_pred = model.predict(X_eval)
    preds = pd.DataFrame(Y_pred, columns=target_vars)
    eval_output = pd.concat([eval_id, preds], 1)
    today = str(dtime.date.today())
    logger.debug("Date:" + today)
    eval_output.to_csv(
        DATA_DIR + '/' + today + '-submission.csv.gz', index=False, float_format='%.5f',
        compression='gzip')
