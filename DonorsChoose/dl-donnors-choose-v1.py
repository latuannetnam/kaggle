#!/usr/bin/python3
# https://www.kaggle.com/c/donorschoose-application-screening
# DonorsChoose.org Application Screening
# Predict whether teachers' project proposals are accepted
# Credit:
#   - https://www.kaggle.com/vlasoff/beginner-s-guide-nn-with-multichannel-input
#   - https://www.kaggle.com/opanichev/lightgbm-and-tf-idf-starter

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

# NLP libraries
import nltk
import nltk.stem as stm
import re
from nltk.tokenize import RegexpTokenizer
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize, pos_tag
from gensim.models import Word2Vec
import spacy
import string
from tqdm import tqdm

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
GLOBAL_DATA_DIR = "/home/latuan/Programming/machine-learning/data"

# VOCAB_SIZE = 8000
VOCAB_SIZE = 4000  # => best
SEQUENCE_LENGTH = 1000  # => best
# SEQUENCE_LENGTH = 2000
OUTPUT_DIM = 300  # use with pretrained word2vec
# OUTPUT_DIM = 50
KERAS_LEARNING_RATE = 0.001
KERAS_N_ROUNDS = 20
# KERAS_BATCH_SIZE = 64  # best
KERAS_BATCH_SIZE = 64
KERAS_NODES = 1024
KERAS_LAYERS = 1
# KERAS_DROPOUT_RATE = 0.5
KERAS_DROPOUT_RATE = 0.2  # => Best
# KERAS_REGULARIZER = KERAS_LEARNING_RATE/10
KERAS_REGULARIZER = 0.04
KERAS_VALIDATION_SPLIT = 0.2
KERAS_EARLY_STOPPING = 3
KERAS_MAXNORM = 3
KERAS_PREDICT_BATCH_SIZE = 4096
# ConvNet
KERAS_FILTERS = 32  # => Best
# KERAS_FILTERS = 4
KERAS_POOL_SIZE = 3  # Best
# KERAS_POOL_SIZE = 1
KERAS_KERNEL_SIZE = 2

# Other keras params
decay = KERAS_LEARNING_RATE / KERAS_N_ROUNDS
nodes = KERAS_NODES
dropout = KERAS_DROPOUT_RATE
KERAS_EMBEDDING = True
N_FOLDS = 5
VERBOSE = True
model_weight_path = OUTPUT_DIR + "/model_weight.h5"
model_path = OUTPUT_DIR + "/pretrained_model.h5"
pretrained_model_path = DATA_DIR + "/pretrained_model.h5"
pretrained_model_weight_path = DATA_DIR + "/model_weight.h5"
# model_path = OUTPUT_DIR + "/model.json"
w2v_weight_path = OUTPUT_DIR + "/w2v_weight.pickle"
random_state = 12343

# ngram_range = 2 will add bi-grams features
NGRAM_RANGE = 1

# Model choice
MODEL_FASTEXT = 1
MODEL_CUDNNLSTM = 2
MODEL_LSTM = 3
MODEL_CNN = 4
MODEL_LSTM_ATTRNN = 5
MODEL_LSTM_HE_ATTRNN = 6
MODEL_INPUT2_DENSE = 10
MODEL_CNN3 = 11
MODEL_CNN2 = 12
MODEL_CUDNNLSTM2 = 13
# Text processing choice
USE_SEQUENCE = True
USE_SPACY = False

# Other params
N_THREADS = psutil.cpu_count()
# disable GPU: export CUDA_VISIBLE_DEVICES=

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


class DonnorsChoose():
    def __init__(self, word2vec=0, model_choice=MODEL_CNN):
        self.word2vec = word2vec
        self.model_choice = model_choice
        self.vocab_size = OUTPUT_DIM
        self.embedding_features = ['project_grade_category',
                                   'project_subject_categories',
                                   'project_subject_subcategories',
                                   'project_title',
                                   'project_essay_1',
                                   'project_essay_2',
                                   'project_essay_3',
                                   'project_essay_4',
                                   'project_resource_summary']
        self.numeric_features = ['teacher_id',
                                 'teacher_prefix',
                                 'school_state',
                                 'project_submitted_datetime',
                                 'teacher_number_of_previously_posted_projects']
        # self.embedding_features = ['name', 'item_description']
        self.special_features = None

    def load_data(self):
        logger.info("Loading data ...")
        # train_data = pd.read_csv(DATA_DIR + "/train.csv", nrows=100)
        # eval_data = pd.read_csv(DATA_DIR + "/test.csv", nrows=100)
        train_data = pd.read_csv(
            DATA_DIR + "/train.csv", low_memory=False)
        eval_data = pd.read_csv(DATA_DIR + "/test.csv",
                                low_memory=False)
        resource_data = pd.read_csv(
            DATA_DIR + "/resources.csv", low_memory=False)

        logger.debug("train size: %s. test size: %s",
                     train_data.shape, eval_data.shape)

        # Group resource by id
        res_data = pd.DataFrame(resource_data[['id', 'quantity', 'price']].groupby('id').agg(
            {
                'quantity': [
                    'sum',
                    'min',
                    'max',
                    'mean',
                    # 'std',
                    # lambda x: len(np.unique(x)),
                ],
                'price': [
                    'sum',
                    'min',
                    'max',
                    'mean',
                    # 'std',
                    # lambda x: len(np.unique(x)),
                ]}
        ))

        # Fallten multi-indexes columns
        res_data.columns = res_data.columns.map('_'.join)
        res_data = res_data.reset_index()
        # Merge train, eval data with resources by id
        train_cb_data = train_data.merge(res_data, on='id', how='left')
        eval_cb_data = eval_data.merge(res_data, on='id', how='left')
        logger.debug("train combined size: %s. test combined size: %s",
                     train_cb_data.shape, eval_cb_data.shape)
        # Combine train + eval data
        self.combine_data = pd.concat(
            [train_cb_data, eval_cb_data], keys=['train', 'eval'])

        logger.debug("combined size: %s", self.combine_data.shape)
        logger.debug("Features: %s", self.combine_data.columns.values)
        # Append features
        self.numeric_features.append(
            ['quantity_sum', 'quantity_min', 'quantity_max', 'quantity_mean'])
        self.numeric_features.append(
            ['price_sum', 'price_min', 'price_max', 'price_mean'])
        print(self.combine_data.head(2))



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

    # set GPU memory
    # KTF.set_session(set_gpu_memory())
    # Fix the issue: The shape of the input to "Flatten" is not fully defined
    # KTF.set_image_dim_ordering('tf')
    object = DonnorsChoose()
    option = 0
    if option == 0:
        object.load_data()
        # object.prepare_data()

    end = time.time() - start
    logger.info("Total time:" + str(end))
    logger.info("Done!")
