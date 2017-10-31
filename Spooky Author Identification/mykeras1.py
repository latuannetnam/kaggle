# Spooky Author Identification
# Share code and discuss insights to identify horror authors from their writings
# https://www.kaggle.com/c/spooky-author-identification
# Credit:
# https://www.kaggle.com/knowledgegrappler/magic-embeddings-keras-a-toy-example
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
from keras.layers import Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras.backend.tensorflow_backend as KTF

# disable GPU: export CUDA_VISIBLE_DEVICES=

pd.options.display.float_format = '{:,.4f}'.format
# Input data files are available in the DATA_DIR directory.
DATA_DIR = "data-temp"
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


def get_keras_data(dataset, maxlen=20):
    X = {
        "stem_input": pad_sequences(dataset.seq_text_stem, maxlen=maxlen)
    }
    return X


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
    train_data.drop(label, 1, inplace=True)
    target_vars = ["EAP", "HPL", "MWS"]

    # train_data.drop(["id", label], axis=1, inplace=True)
    # eval_data.drop(["id"], axis=1, inplace=True)

    logger.debug("Creating sterm words")
    stemmer = stm.SnowballStemmer("english")
    train_data["stem_text"] = train_data.text.apply(lambda x: (" ").join(
        [stemmer.stem(z) for z in re.sub("[^a-zA-Z0-9]", " ", x).split(" ")]))
    eval_data["stem_text"] = eval_data.text.apply(lambda x: (" ").join(
        [stemmer.stem(z) for z in re.sub("[^a-zA-Z0-9]", " ", x).split(" ")]))
    logger.debug(train_data.head(3))

    logger.debug("Tokenizing ...")
    # tok_raw = Tokenizer()
    # tok_raw.fit_on_texts(train_data.text.str.lower())
    tok_stem = Tokenizer()
    tok_stem.fit_on_texts(train_data.stem_text)
    train_data["seq_text_stem"] = tok_stem.texts_to_sequences(
        train_data.stem_text)
    eval_data["seq_text_stem"] = tok_stem.texts_to_sequences(
        eval_data.stem_text)
    logger.debug(train_data["seq_text_stem"][:3])

    # Get training data
    maxlen = 60
    X_train = get_keras_data(train_data, maxlen)
    Y_train = np.array(train_data[target_vars])
    logger.debug("X:" + str(X_train["stem_input"][:3]))
    logger.debug("Y:" + str(Y_train[:3]))
