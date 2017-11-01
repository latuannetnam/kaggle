# Spooky Author Identification
# Share code and discuss insights to identify horror authors from their writings
# https://www.kaggle.com/c/spooky-author-identification
# Credit:
# https://www.kaggle.com/knowledgegrappler/magic-embeddings-keras-a-toy-example
# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
# https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
# System
import datetime as dtime
import time
import logging
import sys
import os
import pickle

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
from keras.layers import Input, Dense, Dropout, BatchNormalization, Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D, LSTM
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

# VOCAB_SIZE = 100
SEQUENCE_LENGTH = 300
OUTPUT_DIM = 300
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
# ConvNet
KERAS_FILTERS = 300
KERAS_KERNEL_SIZE = 2

# Other keras params
decay = KERAS_LEARNING_RATE / KERAS_N_ROUNDS
nodes = KERAS_NODES
dropout = KERAS_DROPOUT_RATE
VERBOSE = True
model_weight_path = DATA_DIR + "/model_weight.h5"
model_path = DATA_DIR + "/model.json"
random_state = 12343

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


def load_pretrained_word_embedding(create=True):
    file = DATA_DIR + "/embeding.pickle"
    if create:
        logger.debug("Loading pre-trained word embbeding ...")    
        # load the whole embedding into memory
        embeddings_index = dict()
        f = open(GLOBAL_DATA_DIR + '/glove.6B.300d.txt')
        for line in f:
            values = line.split()
            word = values[0]
            # print("Word:", word)
            # print(values[1:10])
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
        logger.debug('Loaded ' + str(len(embeddings_index)))

        logger.debug("Create word matrix")
        # create a weight matrix for words in training docs
        embedding_matrix = np.zeros((vocab_size, OUTPUT_DIM))
        for word, i in tokenizer.word_index.items():
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        with open(file, 'wb') as f:
            # Pickle the 'data' dictionary using the highest protocol available.
            pickle.dump(embedding_matrix, f, pickle.HIGHEST_PROTOCOL)
    else:
        logger.debug("Loading from pickle")
        with open(file, 'rb') as f:
            embedding_matrix = pickle.load(f)

    return embedding_matrix


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

    logger.debug("Tokenizing text ..")
    # prepare tokenizer
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_data.text)
    vocab_size = len(tokenizer.word_index) + 1
    logger.debug("vocab size:" + str(vocab_size))
    # integer encode the documents
    encoded_text = tokenizer.texts_to_sequences(train_data.text)
    # print(encoded_text[:3])
    eval_encoded_text = tokenizer.texts_to_sequences(eval_data.text)
    # pad documents to a max length
    X_train = pad_sequences(
        encoded_text, maxlen=SEQUENCE_LENGTH, padding='post')
    X_eval = pad_sequences(
        eval_encoded_text, maxlen=SEQUENCE_LENGTH, padding='post')
    # print(X_train[:3])

    # define the model
    logger.debug("Model definition")
    model = Sequential()
    embedding_matrix = load_pretrained_word_embedding(False)
    embedding_layer = Embedding(vocab_size, OUTPUT_DIM, weights=[
        embedding_matrix], input_length=SEQUENCE_LENGTH, trainable=False)
    model.add(embedding_layer)
    # for i in range(KERAS_LAYERS):
    #     model.add(Conv1D(KERAS_FILTERS, KERAS_KERNEL_SIZE, activation='relu'))
    #     model.add(MaxPooling1D(KERAS_KERNEL_SIZE))
    # model.add(GlobalMaxPooling1D())
    model.add(LSTM(OUTPUT_DIM, activation='relu', dropout=dropout, recurrent_dropout=dropout))
    # model.add(Dropout(dropout, seed=random_state))
    model.add(Dense(KERAS_FILTERS, activation='relu'))
    model.add(Dropout(dropout, seed=random_state))
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

    plt.figure(figsize=(20, 15))
    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # Summmary auc score
    plt.subplot(2, 1, 2)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model acc')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(DATA_DIR + "/history.png")
    logger.info("Done!")
