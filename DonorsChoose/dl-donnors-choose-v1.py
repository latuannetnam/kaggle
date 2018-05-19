#!/usr/bin/python3
# https://www.kaggle.com/c/donorschoose-application-screening
# DonorsChoose.org Application Screening
# Predict whether teachers' project proposals are accepted
# Credit:
#   - https://www.kaggle.com/vlasoff/beginner-s-guide-nn-with-multichannel-input
#   - https://www.kaggle.com/opanichev/lightgbm-and-tf-idf-starter
#   - https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-capsule-networks
#   - https://www.kaggle.com/qinhui1999/deep-learning-is-all-you-need-lb-0-80x/code

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
from keras.layers import concatenate, Input, Dense, Dropout
from keras.layers import BatchNormalization, Flatten, Conv1D, MaxPooling1D
from keras.layers import GlobalMaxPooling1D, LSTM, CuDNNLSTM, Bidirectional
from keras.layers import GlobalAveragePooling1D, Reshape, Dot, Average
from keras.layers import TimeDistributed, GRU, Activation, CuDNNGRU, SpatialDropout1D
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

KERAS_N_ROUNDS = 50
# KERAS_N_ROUNDS = 2
VOCAB_SIZE = 100000
# VOCAB_SIZE = 4000  # => best
# SEQUENCE_LENGTH = 1000  # => best
# SEQUENCE_LENGTH = 2000
SEQUENCE_LENGTH = 300
OUTPUT_DIM = 300  # use with pretrained word2vec
# OUTPUT_DIM = 50
KERAS_LEARNING_RATE = 0.001

# KERAS_BATCH_SIZE = 64  # best
KERAS_BATCH_SIZE = 64
KERAS_NODES = 1024
KERAS_LAYERS = 2
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
# KERAS_POOL_SIZE = 3  # Best
KERAS_POOL_SIZE = 1
KERAS_KERNEL_SIZE = 1

# Other keras params
decay = KERAS_LEARNING_RATE / KERAS_N_ROUNDS
nodes = KERAS_NODES
dropout = KERAS_DROPOUT_RATE
KERAS_EMBEDDING = True
N_FOLDS = 5
# N_FOLDS = 2
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
MODEL_CNN = 2
MODEL_CNN2 = 3
MODEL_CNN3 = 4
MODEL_CNN4 = 15
MODEL_CUDNNLSTM = 5
MODEL_CUDNNLSTM2 = 6
MODEL_LSTM = 7
MODEL_CAPSULE = 8
MODEL_CUDNNGRU = 9
MODEL_INPUT2_DENSE = 10


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


def _get_stem_single(args):
    # https://www.kaggle.com/fmuetsch/keras-nn-with-rec-layers-sentiment-etc-2
    data, index = args
    logger.debug("Stemmer text ..")
    # stemmer = stm.SnowballStemmer("english")
    # stemmer = stm.lancaster.LancasterStemmer()
    stemmer = stm.PorterStemmer()
    data_stem = data.apply(lambda x: (" ").join(
        [stemmer.stem(z) for z in re.sub("[^a-zA-Z0-9]", " ", x).split(" ")]))
    return pd.Series(data_stem, index=index)


def _process_text_row(tokenizer, stop_words, doc):
    tokens = tokenizer.tokenize(doc)
    filtered = [word for word in tokens if word not in stop_words]
    return " ".join(filtered)


def _get_regexp_single(args):
    # https://www.kaggle.com/fmuetsch/keras-nn-with-rec-layers-sentiment-etc-2
    data, index = args
    logger.debug("RegexpTokenizer text ..")
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", ':', ';',
                       '(', ')', '[', ']', '{', '}'])
    new_texts = data.apply(
        lambda x: _process_text_row(tokenizer, stop_words,  x))
    return pd.Series(new_texts, index=index)


def auc(y_true, y_pred):
    # AUC for a binary classifier
    ptas = tf.stack([binary_PTA(y_true, y_pred, k)
                     for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.stack([binary_PFA(y_true, y_pred, k)
                     for k in np.linspace(0, 1, 1000)], axis=0)
    pfas = tf.concat([tf.ones((1,)), pfas], axis=0)
    binSizes = -(pfas[1:] - pfas[:-1])
    s = ptas * binSizes
    return K.sum(s, axis=0)


def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    # PFA, prob false alert for binary classifier
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # N = total number of negative labels
    N = K.sum(1 - y_true)
    # FP = total number of false alerts, alerts from the negative class labels
    FP = K.sum(y_pred - y_pred * y_true)
    return FP / N


def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    # P_TA prob true alerts for binary classifier
    y_pred = K.cast(y_pred >= threshold, 'float32')
    # P = total number of positive labels
    P = K.sum(y_true)
    # TP = total number of correct alerts, alerts from the positive class
    # labels
    TP = K.sum(y_pred * y_true)
    return TP / P


def squash(x, axis=-1):
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale


class Capsule(Layer):
    # https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-capsule-networks
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule,
        # dim_capsule]

        # shape = [None, num_capsule, input_num_capsule]
        b = K.zeros_like(u_hat_vecs[:, :, :, 0])
        for i in range(self.routings):
            # shape = [None, input_num_capsule, num_capsule]
            b = K.permute_dimensions(b, (0, 2, 1))
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


class TokernizedText():
    def __init__(self):
        self.embeddings_index = None

    # Credit:
    # https://github.com/fchollet/keras/blob/master/examples/imdb_fasttext.py
    def create_ngram_set(self, input_list, ngram_value=2):
        """
        Extract a set of n-grams from a list of integers.
        >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
        {(4, 9), (4, 1), (1, 4), (9, 4)}
        >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
        [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
        """
        return set(zip(*[input_list[i:] for i in range(ngram_value)]))

    def add_ngram(self, sequences, token_indice, ngram_range=2):
        """
        Augment the input list of list (sequences) by appending n-grams values.
        Example: adding bi-gram
        >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
        >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
        >>> add_ngram(sequences, token_indice, ngram_range=2)
        [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
        Example: adding tri-gram
        >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
        >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
        >>> add_ngram(sequences, token_indice, ngram_range=3)
        [[1, 3, 4, 5, 1337], [1, 3, 7, 9, 2, 1337, 2018]]
        """
        new_sequences = []
        for input_list in sequences:
            new_list = input_list[:]
            for i in range(len(new_list) - ngram_range + 1):
                for ngram_value in range(2, ngram_range + 1):
                    ngram = tuple(new_list[i:i + ngram_value])
                    if ngram in token_indice:
                        new_list.append(token_indice[ngram])
            new_sequences.append(new_list)

        return new_sequences

    def load_pretrained_word_embedding(self, tokenizer, key, vocab_size, create=True):
        logger.debug("Build embbeding matrix from pre-trained word2vec")
        file = OUTPUT_DIR + "/embeding-" + key + ".pickle"
        if create:
            if self.embeddings_index is None:
                logger.debug("Loading pre-trained word embbeding ...")
                # load the whole embedding into memory
                self.embeddings_index = dict()
                # f = open(GLOBAL_DATA_DIR + '/glove.6B.300d.txt')
                f = open(GLOBAL_DATA_DIR + '/crawl-300d-2M.vec')

                for line in f:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    self.embeddings_index[word] = coefs
                f.close()
                logger.debug('Loaded ' + str(len(self.embeddings_index)))
            logger.debug("Create word matrix")
            # create a weight matrix for words in training docs
            # matrix_size = len(tokenizer.word_index) + 1
            embedding_matrix = np.zeros((vocab_size, OUTPUT_DIM))

            for word, i in tokenizer.word_index.items():
                if i >= vocab_size:
                    continue
                embedding_vector = self.embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector
            with open(file, 'wb') as f:
                # Pickle the 'data' dictionary using the highest protocol
                # available.
                pickle.dump(embedding_matrix, f, pickle.HIGHEST_PROTOCOL)
        else:
            logger.debug("Loading from pickle")
            with open(file, 'rb') as f:
                embedding_matrix = pickle.load(f)
        logger.debug("pre-trained weight size:" + str(embedding_matrix.shape))
        return embedding_matrix

    def tokenize_and_ngram(self, key, texts, word2vec):
        tokenizer = Tokenizer(num_words=VOCAB_SIZE)
        tokenizer.fit_on_texts(texts)
        vocab_size = len(tokenizer.word_index) + 1
        logger.debug("Native vocab size:%d", vocab_size)
        vocab_size = min(vocab_size, VOCAB_SIZE)
        # integer encode the documents
        logger.debug("Text to sequences")
        texts = tokenizer.texts_to_sequences(
            texts)
        # Create n-gram
        if NGRAM_RANGE > 1:
            logger.debug("Creating n-gram ...:" + str(NGRAM_RANGE))
            # Create set of unique n-gram from the training set.
            ngram_set = set()
            for input_list in texts:
                for i in range(2, NGRAM_RANGE + 1):
                    set_of_ngram = self.create_ngram_set(
                        input_list, ngram_value=i)
                    ngram_set.update(set_of_ngram)

            # Dictionary mapping n-gram token to a unique integer.
            # Integer values are greater than max_features in order
            # to avoid collision with existing features.
            start_index = vocab_size + 1
            token_indice = {v: k + start_index for k,
                            v in enumerate(ngram_set)}
            indice_token = {token_indice[k]: k for k in token_indice}

            # max_features is the highest integer that could be found in the
            # dataset.
            vocab_size_new = np.max(list(indice_token.keys())) + 1
            vocab_size = min(vocab_size_new, VOCAB_SIZE * 2)
            logger.debug("New vocab size: %d. Final vocal size:%d",
                         vocab_size_new, vocab_size)

            # Augmenting x_train and x_test with n-grams features
            logger.debug("Adding ngram ...")
            texts = self.add_ngram(
                texts, token_indice, NGRAM_RANGE)

        max_sequence_length = np.amax(
            list(map(len, texts)))
        sequence_length = min(
            max_sequence_length, SEQUENCE_LENGTH)
        logger.debug("Max sequence length:%d. Sequence length:%d",
                     max_sequence_length, sequence_length)
        # pad documents to a max length
        texts = pad_sequences(
            texts, sequence_length)
        logger.debug(key + ":Train shape after padding:" +
                     str(texts.shape))
        logger.debug("Final vocabulary size:%d", vocab_size)

        embedding_matrix = []
        if word2vec == 1:
            logger.debug("Building embbeding matrix ...")
            embedding_matrix = self.load_pretrained_word_embedding(
                tokenizer, key, vocab_size, create=True)
        return texts, embedding_matrix, vocab_size, sequence_length


class DonnorsChoose():
    def __init__(self, word2vec=0, model_choice=MODEL_CNN, model_choice2=MODEL_INPUT2_DENSE):
        self.word2vec = word2vec
        self.model_choice = model_choice
        self.model_choice2 = model_choice2
        self.vocab_size = OUTPUT_DIM
        self.scaler = StandardScaler()
        self.text_features = [
            'project_title',
            'project_resource_summary',
            'project_essay_1',
            'project_essay_2',
            'project_essay_3',
            'project_essay_4',
        ]

        self.embedding_features = [
            'project_title',
            'project_essay',
            'project_resource_summary'
            # 'text'
        ]
        # self.embedding_features = []
        self.category_features = ['teacher_id',
                                  'teacher_prefix',
                                  'school_state',
                                  'project_grade_category',
                                  'project_subject_categories',
                                  'project_subject_subcategories',
                                  ]
        self.numeric_features = [
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
                    'count',
                    'min',
                    'max',
                    'mean',
                    'std',
                    # lambda x: len(np.unique(x)),
                ],
                'price': [
                    'sum',
                    'count',
                    'min',
                    'max',
                    'mean',
                    'std',
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
        self.numeric_features.extend(
            ['quantity_sum', 'quantity_min', 'quantity_max', 'quantity_mean'])
        self.numeric_features.extend(
            ['price_sum', 'price_min', 'price_max', 'price_mean'])
        # print(self.combine_data.head(2))

    def prepare_data(self):
        logger.debug("Data features preparing...")
        # Handle missing value
        self.combine_data = self.handle_missing(self.combine_data)
        # Combine project essay
        logger.debug("Combining project essay...")
        self.combine_data.loc[:, 'project_essay'] = self.combine_data.apply(
            lambda row: ' '.join([
                str(row['project_essay_1']),
                str(row['project_essay_2']),
                str(row['project_essay_3']),
                str(row['project_essay_4']),
            ]), axis=1)

        # Combine all text features
        logger.debug("Combining all text features...")
        self.combine_data.loc[:, 'text'] = self.combine_data.apply(
            lambda row: ' '.join([
                str(row[col]) for col in self.text_features
            ]), axis=1)

        for key in self.embedding_features:
            logger.debug("Text processing for %s", key)
            self.combine_data[key] = self.preprocess_text(
                self.combine_data[key], preprocess=4)

        # Tokenize text data
        self.tokenize_and_ngram()

        # Process datetime feature
        self.convert_datetime()
        # Process categorial features
        for key in self.category_features:
            logger.debug("Categorying for %s", key)
            self.combine_data[key] = pd.factorize(self.combine_data[key])[0]

        # Append category features to numberic features
        # self.numeric_features.extend(self.category_features)
        logger.debug("Categorial features: %s", self.category_features)
        logger.debug("Numerical features: %s", self.numeric_features)

        # Split train and test set
        logger.debug("Split train and eval data set")
        self.train_data = self.combine_data.loc['train'].copy()
        self.eval_data = self.combine_data.loc['eval'].copy()
        del self.combine_data

        # CREATE TARGET VARIABLE
        self.target = self.train_data[['project_is_approved']].values
        self.eval_id = self.eval_data['id']
        logger.debug("train size:" + str(self.train_data.shape))
        logger.debug("test size:" + str(self.eval_data.shape))

    def handle_missing(self, dataset):
        for key in self.text_features:
            dataset[key].fillna(value=" ", inplace=True)
        return (dataset)

    def convert_datetime(self):
        logger.info("Convert datetime ...")
        data = self.combine_data
        data.loc[:, 'datetime_obj'] = pd.to_datetime(
            data['project_submitted_datetime'])
        data.loc[:, 'project_year'] = data['datetime_obj'].dt.year
        data.loc[:, 'project_month'] = data['datetime_obj'].dt.month
        data.loc[:, 'project_day'] = data['datetime_obj'].dt.day
        # data.loc[:, 'project_weekday'] = data['datetime_obj'].dt.weekday
        # data.loc[:, 'project_dayofyear'] = data['datetime_obj'].dt.dayofyear
        # data.loc[:, 'project_hour'] = data['datetime_obj'].dt.hour
        # data.loc[:, 'project_whour'] = data['project_weekday'] * \
        #     24 + data['project_hour']
        # data.loc[:, 'project_minute'] = data['datetime_obj'].dt.minute
        self.numeric_features.extend(
            ['project_year', 'project_month', 'project_day'])

    def get_stem(self, data, index):
        p = Pool(processes=N_THREADS)
        n = math.ceil(len(data) / N_THREADS)
        stems = p.map(_get_stem_single, [
            (data[i:i + n], index[i:i + n]) for i in range(0, len(data), n)])
        # return np.array(flatten(stems))
        # return stems
        return pd.concat(stems)

    def get_regexp(self, data, index):
        p = Pool(processes=N_THREADS)
        n = math.ceil(len(data) / N_THREADS)
        texts = p.map(_get_regexp_single, [
            (data[i:i + n], index[i:i + n]) for i in range(0, len(data), n)])
        return pd.concat(texts)

    def process_text_row(self, tokenizer, stop_words, doc):
        tokens = tokenizer.tokenize(doc)
        filtered = [word for word in tokens if word not in stop_words]
        return " ".join(filtered)

    def preprocess_text(self, texts, preprocess=1):
        new_texts = texts
        if preprocess == 1:
            logger.debug("Preprocessing with STEM ...")
            stemmer = stm.PorterStemmer()
            new_texts = texts.apply(lambda x: (" ").join(
                [stemmer.stem(z) for z in re.sub("[^a-zA-Z0-9]", " ", x).split(" ")]))
        elif preprocess == 2:
            logger.debug("Preprocessing with STEM multi-thread...")
            new_texts = self.get_stem(texts, texts.index)
        elif preprocess == 3:
            logger.debug("Preprocessing with RegexpTokenizer ...")
            tokenizer = RegexpTokenizer(r'\w+')
            stop_words = set(stopwords.words('english'))
            stop_words.update(['.', ',', '"', "'", ':', ';',
                               '(', ')', '[', ']', '{', '}'])
            new_texts = texts.apply(
                lambda x: self.process_text_row(tokenizer, stop_words,  x))
        elif preprocess == 4:
            logger.debug("Preprocessing with RegexpTokenizer multi-thread...")
            new_texts = self.get_regexp(texts, texts.index)

        logger.debug("Done Preprocessing ")
        return new_texts

    def tokenize_and_ngram(self):
        self.vocab_size = {}
        self.sequence_length = {}
        self.embedding_matrix = {}
        object = TokernizedText()
        for key in self.embedding_features:
            logger.debug("Tokenizing for " + key)
            self.combine_data[key], self.embedding_matrix[key], self.vocab_size[key], self.sequence_length[key] = object.tokenize_and_ngram(
                key, self.combine_data[key], self.word2vec)
        del object

    def buil_embbeding_layer(self, name, vocab_size, output_dim, input_length, word2vec=0):
        if word2vec > 0:
            embedding_layer = Embedding(vocab_size, output_dim, weights=[
                self.embedding_matrix[name]], input_length=input_length, trainable=True)
        else:
            embedding_layer = Embedding(
                vocab_size, output_dim, input_length=input_length, trainable=True)
        return embedding_layer

    def model_fasttext(self, name, vocab_size, output_dim, input_length, word2vec=0):
        logger.debug("Building FastText model for %s ...", name)
        embbeding_input = Input(shape=(None,), name=name)
        embedding_layer = self.buil_embbeding_layer(name,
                                                    vocab_size, output_dim, input_length, word2vec)(embbeding_input)
        model = Dropout(dropout, seed=random_state)(embedding_layer)
        model = GlobalAveragePooling1D()(model)
        model = Dropout(dropout, seed=random_state)(model)
        return embbeding_input, model

    def model_cnn(self, name, vocab_size, output_dim, input_length, word2vec=0):
        logger.debug("Building CNN model for %s ...", name)
        embbeding_input = Input(shape=(None,), name=name)
        embedding_layer = self.buil_embbeding_layer(name,
                                                    vocab_size, output_dim, input_length, word2vec)(embbeding_input)
        model = Dropout(dropout, seed=random_state)(embedding_layer)
        for i in range(KERAS_LAYERS):
            model = Conv1D(output_dim, KERAS_KERNEL_SIZE,
                           activation='relu')(model)
            model = MaxPooling1D(pool_size=KERAS_POOL_SIZE)(model)
            model = Dropout(dropout, seed=random_state)(model)
        model = GlobalMaxPooling1D()(model)
        model = Dropout(dropout, seed=random_state)(model)
        # model = BatchNormalization()(model)
        return embbeding_input, model

    # https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras/blob/master/sentiment_cnn.py
    def model_cnn2(self, name, vocab_size, output_dim, input_length, word2vec=0):
        logger.debug("Building CNN2 model for %s ...", name)
        embbeding_input = Input(shape=(None,), name=name)
        embedding_layer = self.buil_embbeding_layer(name,
                                                    vocab_size, output_dim, input_length, word2vec)(embbeding_input)
        model = Dropout(dropout, seed=random_state)(embedding_layer)
        filter_sizes = [1, 2]  # => Best
        # filter_sizes = [2, 3, 4, 5]
        conv_blocks = []
        for fsz in filter_sizes:
            conv = Conv1D(filters=KERAS_FILTERS,
                          kernel_size=fsz,
                          padding="valid",
                          activation="relu",
                          strides=1)(model)
            conv = MaxPooling1D(pool_size=KERAS_POOL_SIZE)(conv)
            conv = Flatten()(conv)
            # conv = Dropout(dropout, seed=random_state)(conv)
            conv_blocks.append(conv)
        model_merge = concatenate(conv_blocks)
        model = Dropout(dropout, seed=random_state)(model_merge)
        return embbeding_input, model

    # https://richliao.github.io/supervised/classification/2016/11/26/textclassifier-convolutional/
    def model_cnn3(self, name, vocab_size, output_dim, input_length, word2vec=0):
        logger.debug("Building CNN3 model for %s ...", name)
        embbeding_input = Input(shape=(None,), name=name)
        embedding_layer = self.buil_embbeding_layer(name,
                                                    vocab_size, output_dim, input_length, word2vec)(embbeding_input)
        model = Dropout(dropout, seed=random_state)(embedding_layer)
        convs = []
        filter_sizes = [3, 4, 5]
        for fsz in filter_sizes:
            l_conv = Conv1D(filters=128, kernel_size=fsz,
                            activation='relu')(model)
            l_pool = MaxPooling1D(5)(l_conv)
            convs.append(l_pool)
        l_merge = concatenate(convs)
        l_merge = Dropout(dropout, seed=random_state)(l_merge)
        l_cov1 = Conv1D(128, 5, activation='relu')(l_merge)
        l_pool1 = MaxPooling1D(5)(l_cov1)
        # l_cov2 = Conv1D(128, 5, activation='relu')(l_pool1)
        # l_pool2 = MaxPooling1D(30)(l_cov2)
        # https://github.com/fchollet/keras/issues/1592 => If error if Flatten,
        # refer to this issue, may be pooling size is too large
        l_flat = Flatten()(l_pool1)
        l_flat = Dropout(dropout, seed=random_state)(l_flat)
        l_dense = Dense(128, activation='relu')(l_flat)
        l_dense = Dropout(dropout, seed=random_state)(l_dense)
        return embbeding_input, l_dense

    def model_cnn4(self, name, vocab_size, output_dim, input_length, word2vec=0):
        logger.debug("Building CNN model v4 for %s ...", name)
        embbeding_input = Input(shape=(None,), name=name)
        embedding_layer = self.buil_embbeding_layer(name,
                                                    vocab_size, output_dim, input_length, word2vec)(embbeding_input)
        model = Dropout(dropout, seed=random_state)(embedding_layer)
        model = Conv1D(filters=KERAS_FILTERS, kernel_size=KERAS_KERNEL_SIZE,
                       activation='relu')(model)
        model = MaxPooling1D(pool_size=KERAS_POOL_SIZE)(model)
        model = Dropout(dropout, seed=random_state)(model)
        model = GlobalMaxPooling1D()(model)
        model = Dropout(dropout, seed=random_state)(model)
        # model = BatchNormalization()(model)
        return embbeding_input, model

    def model_cudnnlstm(self, name, vocab_size, output_dim, input_length, word2vec=0):
        logger.debug("Building CuDNN LSTM model for %s ...", name)
        embbeding_input = Input(shape=(None,), name=name)
        embedding_layer = self.buil_embbeding_layer(name,
                                                    vocab_size, output_dim, input_length, word2vec)(embbeding_input)
        model = Dropout(dropout, seed=random_state)(embedding_layer)
        # LSTM
        model = CuDNNLSTM(output_dim)(model)
        model = Dropout(dropout, seed=random_state)(model)
        # model = BatchNormalization()(model)
        return embbeding_input, model

    def model_cudnnlstm2(self, name, vocab_size, output_dim, input_length, word2vec=0):
        logger.debug(
            "Building Bidirectional CuDNN LSTM model for %s ...", name)
        embbeding_input = Input(shape=(None,), name=name)
        embedding_layer = self.buil_embbeding_layer(name,
                                                    vocab_size, output_dim, input_length, word2vec)(embbeding_input)
        model = Dropout(dropout, seed=random_state)(embedding_layer)
        # LSTM
        model = Bidirectional(
            CuDNNLSTM(output_dim // 2, return_sequences=True))(model)
        model = GlobalMaxPooling1D()(model)
        model = Dropout(dropout, seed=random_state)(model)
        # model = BatchNormalization()(model)
        return embbeding_input, model

    def model_cudnngru(self, name, vocab_size, output_dim, input_length, word2vec=0):
        logger.debug(
            "Building Bidirectional CuDNN GRU model for %s ...", name)
        embbeding_input = Input(shape=(None,), name=name)
        embedding_layer = self.buil_embbeding_layer(name,
                                                    vocab_size, output_dim, input_length, word2vec)(embbeding_input)
        model = Dropout(dropout, seed=random_state)(embedding_layer)
        # LSTM
        model = Bidirectional(
            CuDNNGRU(output_dim // 2, return_sequences=True))(model)
        model = Conv1D(output_dim, KERAS_KERNEL_SIZE, activation="relu")(model)
        model = GlobalMaxPooling1D()(model)
        model = Dropout(dropout, seed=random_state)(model)
        # model = BatchNormalization()(model)
        return embbeding_input, model

    def model_lstm(self, input_length):
        logger.debug("Building LSTM model for %s ...", name)
        embbeding_input = Input(shape=(None,))
        embedding_layer = self.buil_embbeding_layer(name,
                                                    input_length)(embbeding_input)
        model = Dropout(dropout, seed=random_state)(embedding_layer)
        # for i in range(KERAS_LAYERS):
        #     model = Conv1D(OUTPUT_DIM, KERAS_KERNEL_SIZE,
        #                    activation='relu')(model)
        #     model = MaxPooling1D(pool_size=KERAS_POOL_SIZE)(model)
        #     model = Dropout(dropout, seed=random_state)(model)
        # LSTM
        model = LSTM(OUTPUT_DIM, activation='relu',
                     dropout=dropout, recurrent_dropout=dropout)(model)
        model = BatchNormalization()(model)
        return embbeding_input, model

    def model_capsule(self, name, vocab_size, output_dim, input_length, word2vec=0):
        # https://www.kaggle.com/fizzbuzz/beginner-s-guide-to-capsule-networks
        logger.debug(
            "Building Capsule model for %s ...", name)
        gru_len = 128
        routings = 5
        num_capsule = 10
        dim_capsule = 16
        embbeding_input = Input(shape=(None,), name=name)
        embedding_layer = self.buil_embbeding_layer(name,
                                                    vocab_size, output_dim, input_length, word2vec)(embbeding_input)
        embedding_layer = SpatialDropout1D(dropout)(embedding_layer)
        x = Bidirectional(
            GRU(gru_len, activation='relu', dropout=dropout, recurrent_dropout=dropout, return_sequences=True))(
            embedding_layer)
        capsule = Capsule(num_capsule=num_capsule, dim_capsule=dim_capsule, routings=routings,
                          share_weights=True)(x)
        capsule = Flatten()(capsule)
        model = Dropout(dropout)(capsule)
        return embbeding_input, model

    def model_input2_dense(self, name, data):
        logger.debug("Building dense model from numerical features ...")
        n_features = data.shape[1]
        logger.debug("Num features of input:" + str(n_features))
        feature_input = Input(shape=(n_features,), name=name)
        model = Dropout(KERAS_DROPOUT_RATE, seed=random_state)(feature_input)
        # model = feature_input
        # nodes = OUTPUT_DIM
        nodes = min(n_features * 2, OUTPUT_DIM)
        layers = max(KERAS_LAYERS, 1)
        for i in range(layers):
            model = (Dense(nodes,
                           activation='relu', kernel_constraint=keras.constraints.maxnorm(KERAS_MAXNORM)))(model)
            model = Dropout(KERAS_DROPOUT_RATE, seed=random_state)(model)
            model = BatchNormalization()(model)
            nodes = int(nodes // 2)
            if nodes < 2:
                nodes = 2
        # logger.debug("Input2 model shape:" + str(model.shape))
        return feature_input, model

    def build_model(self, X_train):
        logger.debug("Model definition")

        if self.model_choice == MODEL_FASTEXT:
            model = self.model_fasttext

        elif self.model_choice == MODEL_CUDNNLSTM:
            model = self.model_cudnnlstm

        elif self.model_choice == MODEL_CUDNNLSTM2:
            model = self.model_cudnnlstm2

        elif self.model_choice == MODEL_LSTM:
            model = self.model_lstm

        elif self.model_choice == MODEL_CNN:
            model = self.model_cnn

        elif self.model_choice == MODEL_CNN2:
            model = self.model_cnn2

        elif self.model_choice == MODEL_CNN3:
            model = self.model_cnn3

        elif self.model_choice == MODEL_CNN4:
            model = self.model_cnn4

        elif self.model_choice == MODEL_CAPSULE:
            model = self.model_capsule

        elif self.model_choice == MODEL_CUDNNGRU:
            model = self.model_cudnngru

        # Build models for each features in dataset
        inputs = []
        in_models = []
        for key in sorted(X_train):
            logger.debug("Building model for %s", key)
            if key in self.embedding_features:
                input, in_model = model(
                    key, self.vocab_size[key], OUTPUT_DIM, self.sequence_length[key], word2vec=1)
                # inputs.append(input)
                # in_models.append(in_model)
            elif key == 'num_vars' or key == 'cat_vars':
                if self.model_choice2 == MODEL_INPUT2_DENSE:
                    logger.debug("Bulding dense layer for %s", key)
                    input, in_model = self.model_input2_dense(
                        key, X_train[key])
                else:
                    vocab_size = int(np.max(X_train[key]) + 1)
                    sequence_length = X_train[key].shape[1]
                    logger.debug("Vocab size for num_vars:" + str(vocab_size) +
                                 " . Sequence length:" + str(sequence_length))
                    input, in_model = model(
                        key, vocab_size, sequence_length, sequence_length, word2vec=0)

            inputs.append(input)
            in_models.append(in_model)

        if len(in_models) > 1:
            model_all = concatenate(in_models)
            model_all = Dropout(KERAS_DROPOUT_RATE,
                                seed=random_state)(model_all)
        else:
            model_all = in_models[0]

        # Output layer
        out_model = Dense(1,
                          activation="sigmoid")(model_all)
        self.model = Model(
            inputs=inputs, outputs=out_model)
        # compile the model
        optimizer = Adam(lr=KERAS_LEARNING_RATE, decay=decay)
        # optimizer = RMSprop(lr=KERAS_LEARNING_RATE, decay=decay)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy',
                           metrics=['accuracy', auc])
        # summarize the model
        print(self.model.summary())
        # Plot the model
        plot_model(self.model, show_shapes=True,
                   to_file=OUTPUT_DIR + '/model.png')

    def load_model(self):
        logger.debug("Build and load pre-train model")
        self.build_model()
        # self.model = load_model(pretrained_model_path)
        self.model.load_weights(pretrained_model_weight_path)

    def get_keras_data(self, dataset):
        X = {}
        for key in self.embedding_features:
            X[key] = dataset[key].values
        if self.model_choice2 is not None:
            X['num_vars'] = dataset[self.numeric_features].values
            X['cat_vars'] = dataset[self.category_features].values
        return X

    def train_single_model(self):
        logger.info("Training for single model ...")
        logger.debug(" Spliting train and test set...")
        self.build_model(self.get_keras_data(self.train_data))
        d_train, d_valid, Y_train, Y_test = train_test_split(
            self.train_data, self.target, test_size=KERAS_VALIDATION_SPLIT, shuffle=False, random_state=1234)
        X_train = self.get_keras_data(d_train)
        X_test = self.get_keras_data(d_valid)
        # Scale numeric features
        logger.debug("Scaling mumeric features")
        X_train['num_vars'] = self.scaler.fit_transform(X_train['num_vars'])
        X_test['num_vars'] = self.scaler.transform(X_test['num_vars'])

        for key in X_train:
            logger.debug("Key: %s, shape: %s", key, str(X_train[key].shape))

        # Use Early-Stopping
        callback_early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_auc', patience=KERAS_EARLY_STOPPING, verbose=VERBOSE, mode='max')
        callback_tensorboard = keras.callbacks.TensorBoard(
            log_dir=OUTPUT_DIR + '/tensorboard', histogram_freq=1, batch_size=KERAS_BATCH_SIZE,
            write_graph=True, write_grads=True, write_images=False)
        callback_checkpoint = keras.callbacks.ModelCheckpoint(
            model_weight_path, monitor='val_auc', verbose=VERBOSE, save_best_only=True, mode='max')
        lr_reduced = keras.callbacks.ReduceLROnPlateau(monitor='val_auc',
                                                       factor=0.1,
                                                       patience=KERAS_EARLY_STOPPING - 2,
                                                       verbose=VERBOSE,
                                                       epsilon=1e-4,
                                                       mode='max')
        # Compute class weigth to balance label
        cw = class_weight.compute_class_weight(
            'balanced', np.unique(self.target[:, 0]), self.target[:, 0])
        class_weight_dict = dict(enumerate(cw))
        logger.debug("Training ...")
        start = time.time()
        # Training model
        self.history = self.model.fit([X_train[key] for key in sorted(X_train)], Y_train,
                                      validation_data=(
            [X_test[key] for key in sorted(X_test)], Y_test),
            batch_size=KERAS_BATCH_SIZE,
            epochs=KERAS_N_ROUNDS,
            callbacks=[
            # callback_tensorboard,
            callback_early_stopping,
            callback_checkpoint,
            lr_reduced
        ],
            class_weight=class_weight_dict,
            verbose=VERBOSE
        )
        end = time.time() - start
        logger.debug("Train time:" + str(end))
        # load best model
        logger.info("Loading best model ...")
        self.model.load_weights(model_weight_path)
        # metric = self.history.history['val_keras_rmse']
        # loss_metric = self.history.history['val_loss']
        # best_loss = min(loss_metric)
        # best_loss_index = len(loss_metric) - 1 - loss_metric[::-1].index(best_loss)
        # best_metric = metric[best_loss_index]
        # logger.debug('Best loss:' + str(best_loss))
        # logger.debug('Best metric:' + str(best_metric))
        # logger.debug('Min metric:' + str(min(metric)))
        logger.debug('Best metric:' + str(callback_early_stopping.best))
        logger.debug(
            'Best round:' + str(callback_early_stopping.stopped_epoch + 1 - KERAS_EARLY_STOPPING))
        # self.model.save(model_path)

    def train_kfold_single(self):
        logger.info("Train Kfold for single model")

        logger.debug("Prepare training data ...")
        X = self.get_keras_data(self.train_data)
        X_Kfold = X[self.embedding_features[0]]
        Y = self.target
        T = self.get_keras_data(self.eval_data)
        T_Kfold = T[self.embedding_features[0]]
        Y_eval = np.zeros((T_Kfold.shape[0], Y.shape[1], N_FOLDS))
        logger.debug("Y eval size:" + str(Y_eval.shape))
        total_time = 0
        total_metric = 0
        total_best_round = 0
        self.history_total = []
        # build model and train
        self.build_model(X)
        # Save initilize model weigth for reseting weigth after each loop
        model_init_weights = self.model.get_weights()
        # Compute class weigth to balance label
        cw = class_weight.compute_class_weight(
            'balanced', np.unique(self.target[:, 0]), self.target[:, 0])
        class_weight_dict = dict(enumerate(cw))
        # Setup KFold
        kfolds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=321)
        for j, (train_idx, test_idx) in enumerate(kfolds.split(X_Kfold)):
            logger.debug("Round:" + str(j + 1))
            start = time.time()
            X_train = {}
            X_test = {}
            # Split train and test data
            for key in sorted(X):
                X_train[key] = X[key][train_idx]
                X_test[key] = X[key][test_idx]
                logger.debug(
                    key + ":Train size:" + str(X_train[key].shape) + " .Test size:" + str(X_test[key].shape))

            Y_train = Y[train_idx]
            Y_test = Y[test_idx]

            # Use Early-Stopping
            callback_early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_auc', patience=KERAS_EARLY_STOPPING, verbose=VERBOSE, mode='max')
            callback_tensorboard = keras.callbacks.TensorBoard(
                log_dir=OUTPUT_DIR + '/tensorboard', histogram_freq=1, batch_size=KERAS_BATCH_SIZE,
                write_graph=True, write_grads=True, write_images=False)
            callback_checkpoint = keras.callbacks.ModelCheckpoint(
                model_weight_path, monitor='val_auc', verbose=VERBOSE, save_best_only=True, mode='max')
            lr_reduced = keras.callbacks.ReduceLROnPlateau(monitor='val_auc',
                                                           factor=0.1,
                                                           patience=KERAS_EARLY_STOPPING - 2,
                                                           verbose=VERBOSE,
                                                           epsilon=1e-4,
                                                           mode='max')
            # Training
            history = self.model.fit([X_train[key] for key in sorted(X_train)],
                                     Y_train,
                                     validation_data=(
                [X_test[key] for key in sorted(X_test)], Y_test),
                batch_size=KERAS_BATCH_SIZE,
                epochs=KERAS_N_ROUNDS,
                callbacks=[
                # callback_tensorboard,
                callback_early_stopping,
                callback_checkpoint,
                lr_reduced
            ],
                class_weight=class_weight_dict,
                verbose=VERBOSE
            )

            end = time.time() - start
            total_time = total_time + end
            self.history_total.append(history)
            # load best model
            logger.info("Loading best model ...")
            self.model.load_weights(model_weight_path)
            metric = callback_early_stopping.best
            total_metric = total_metric + metric
            logger.debug('Best metric:' + str(metric))
            best_round = callback_early_stopping.stopped_epoch + 1 - KERAS_EARLY_STOPPING
            total_best_round = total_best_round + best_round
            logger.debug('Best round:' + str(best_round))
            # Predict
            logger.debug("Saving Y_eval for round:" + str(j + 1))
            Y_eval[:, :, j] = self.model.predict([T[key] for key in sorted(T)])
            # Reset weigth
            logger.debug("Reset model weights")
            self.model.set_weights(model_init_weights)
            logger.debug("Done training for round:" + str(j + 1) +
                         " time:" + str(end) + "/" + str(total_time))

        logger.debug("Avg metric:" + str(total_metric / (j + 1)))
        logger.debug("Avg best round:" + str(total_best_round / (j + 1)))
        Y_eval_total = Y_eval.mean(2)
        logger.debug("Y eval size:" + str(Y_eval_total.shape))
        logger.debug("Total training time:" + str(total_time))
        return Y_eval_total

    def predict_data(self, Y_pred=None):
        # PREDICTION
        logger.debug("Prediction")
        if Y_pred is None:
            X_eval = self.get_keras_data(self.eval_data)
            X_eval['num_vars'] = self.scaler.transform(X_eval['num_vars'])
            Y_pred = self.model.predict([X_eval[key]
                                         for key in sorted(X_eval)])
        eval_output = pd.read_csv(DATA_DIR + '/sample_submission.csv')
        eval_output.loc[:, 'project_is_approved'] = Y_pred
        today = str(dtime.date.today())
        logger.debug("Date:" + today)
        eval_output.to_csv(
            DATA_DIR + '/' + today + '-submission.csv.gz', index=False, float_format='%.5f',
            compression='gzip')

    def plot_history(self):
        history = self.history
        plt.figure(figsize=(20, 15))
        plt.subplot(2, 1, 1)
        plt.plot(history.history['auc'])
        plt.plot(history.history['val_auc'])
        plt.title('model AUC')
        plt.ylabel('AUC')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # Summmary loss score
        plt.subplot(2, 1, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(DATA_DIR + "/history.png")

    def plot_kfold_history(self):
        history = self.history_total
        plt.figure(figsize=(20, 15))
        plt.subplot(2, 1, 1)
        plt.title('model AUC')
        plt.ylabel('AUC')
        plt.xlabel('epoch')
        for i, history in enumerate(self.history_total):
            plt.plot(history.history['auc'],
                     label='train-' + str(i + 1))
            plt.plot(history.history['val_auc'],
                     label='test-' + str(i + 1))
        plt.legend()
        # Summmary loss score
        plt.subplot(2, 1, 2)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        for i, history in enumerate(self.history_total):
            plt.plot(history.history['loss'],
                     label='train-' + str(i + 1))
            plt.plot(history.history['val_loss'],
                     label='test-' + str(i + 1))
        plt.legend()
        plt.savefig(DATA_DIR + "/history.png")


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
    # object = DonnorsChoose(word2vec=1, model_choice=MODEL_FASTEXT,
    #                        model_choice2=MODEL_INPUT2_DENSE)
    object = DonnorsChoose(
        word2vec=1, model_choice=MODEL_CUDNNLSTM,
        model_choice2=MODEL_INPUT2_DENSE)
    # object = DonnorsChoose(
    #     word2vec=1, model_choice=MODEL_CUDNNGRU,
    #     model_choice2=MODEL_INPUT2_DENSE)
    # object = DonnorsChoose(
    #     word2vec=1, model_choice=MODEL_CNN4, model_choice2=MODEL_INPUT2_DENSE)
    # object = DonnorsChoose(word2vec=1, model_choice=MODEL_CAPSULE,
    #                        model_choice2=MODEL_INPUT2_DENSE)
    # object = DonnorsChoose(
    #     word2vec=1, model_choice=MODEL_FASTEXT, model_choice2=MODEL_FASTEXT)
    option = 1
    if option == 0:
        object.load_data()
        object.prepare_data()
    if option == 1:
        object.load_data()
        object.prepare_data()
        object.train_single_model()
        object.predict_data()
        object.plot_history()
    elif option == 2:
        object.load_data()
        object.prepare_data()
        Y_pred = object.train_kfold_single()
        object.predict_data(Y_pred)
        object.plot_kfold_history()

    end = time.time() - start
    logger.info("Total time:" + str(end))
    logger.info("Done!")
