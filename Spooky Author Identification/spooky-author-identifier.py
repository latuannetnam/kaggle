# Spooky Author Identification
# Share code and discuss insights to identify horror authors from their writings
# https://www.kaggle.com/c/spooky-author-identification
# Credit:
# https://www.kaggle.com/knowledgegrappler/magic-embeddings-keras-a-toy-example
# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
# https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
# https://martinbel.github.io/fast-text.html
# http://debajyotidatta.github.io/nlp/deep/learning/word-embeddings/2016/09/28/fast-text-and-skip-gram/
# http://ben.bolte.cc/blog/2016/gensim.html
# https://www.bonaccorso.eu/2017/08/07/twitter-sentiment-analysis-with-gensim-word2vec-and-keras-convolutional-networks/
# https://www.kaggle.com/enerrio/scary-nlp-with-spacy-and-keras
# https://richliao.github.io/supervised/classification/2016/12/26/textclassifier-RNN/
# https://richliao.github.io/supervised/classification/2016/11/26/textclassifier-convolutional/
# https://github.com/synthesio/hierarchical-attention-networks/blob/master/model.py
# https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2

# System
import datetime as dtime
import time
import logging
import sys
import os
import pickle
import glob

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
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopword2s
from sklearn import decomposition

# Keras
import keras
from keras.models import Sequential, Model
from keras.layers import concatenate, Input, Dense, Dropout, BatchNormalization, Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D, LSTM, CuDNNLSTM, Bidirectional, GlobalAveragePooling1D, Reshape, Dot, Average, TimeDistributed, GRU, Activation
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


# Constants
# Input data files are available in the DATA_DIR directory.
DATA_DIR = "data-temp"
GLOBAL_DATA_DIR = "/home/latuan/Programming/machine-learning/data"
EXTRA_DATA_DIR = "extra-data"
LOVECRAFT_DIR = EXTRA_DATA_DIR + "/corpus/lovecraft"
POE_DIR = EXTRA_DATA_DIR + "/corpus/poe"
SHELLEY_DIR = EXTRA_DATA_DIR + "/corpus/shelley"
# VOCAB_SIZE = 100
SEQUENCE_LENGTH = 500
OUTPUT_DIM = 300  # use with pretrained word2vec
# OUTPUT_DIM = 500
KERAS_LEARNING_RATE = 0.001
KERAS_N_ROUNDS = 200
KERAS_BATCH_SIZE = 64
KERAS_NODES = 1024
KERAS_LAYERS = 1
KERAS_DROPOUT_RATE = 0.5
# KERAS_REGULARIZER = KERAS_LEARNING_RATE/10
KERAS_REGULARIZER = 0.04
KERAS_VALIDATION_SPLIT = 0.2
KERAS_EARLY_STOPPING = 3
KERAS_MAXNORM = 3
KERAS_PREDICT_BATCH_SIZE = 1024
# ConvNet
# KERAS_FILTERS = OUTPUT_DIM
KERAS_FILTERS = 32
KERAS_KERNEL_SIZE = 2
KERAS_POOL_SIZE = 3

# Other keras params
decay = KERAS_LEARNING_RATE / KERAS_N_ROUNDS
nodes = KERAS_NODES
dropout = KERAS_DROPOUT_RATE
KERAS_EMBEDDING = True
N_FOLDS = 5
VERBOSE = True
model_weight_path = DATA_DIR + "/model_weight.h5"
model_path = DATA_DIR + "/model.json"
w2v_weight_path = DATA_DIR + "/w2v_weight.pickle"
random_state = 12343

# ngram_range = 2 will add bi-grams features
ngram_range = 2

# Model choice
MODEL_FASTEXT = 1
MODEL_CUDNNLSTM = 2
MODEL_LSTM = 3
MODEL_CNN = 4
MODEL_LSTM_ATTRNN = 5
MODEL_LSTM_HE_ATTRNN = 6
MODEL_INPUT2_DENSE = 10
# Text processing choice
USE_SEQUENCE = True
USE_SPACY = False
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


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


def id_generator(size=5, chars=string.digits):
    return 'id' + ''.join(rn.choice(chars) for _ in range(size))


# https://richliao.github.io/supervised/classification/2016/12/26/textclassifier-RNN/
# https://github.com/huajianjiu/textClassifier/blob/master/textClassifierHATT.py
class AttLayer(Layer):
    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttLayer, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = K.dot(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        # ait = K.dot(uit, self.u)  # replace this
        mul_a = uit * self.u  # with this
        ait = K.sum(mul_a, axis=2)  # and this

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


# https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2
class AttentionWithContext(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.

    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.

    Note: The layer has been tested with Keras 2.0.6

    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        # ait = K.dot(uit, self.u)
        ait = dot_product(uit, self.u)
        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


class SpacyPreprocess():
    def __init__(self):
        self.nlp = spacy.load('en')
        self.punctuations = string.punctuation

    def cleanup_text(self, docs):
        logger.debug("Cleanup text ...")
        texts = []
        for doc in docs:
            doc = self.nlp(doc)
            tokens = [tok.lemma_.lower().strip()
                      for tok in doc if tok.lemma_ != '-PRON-']
            tokens = [
                tok for tok in tokens if tok not in stopword2s and tok not in self.punctuations]
            tokens = ' '.join(tokens)
            texts.append(tokens)
        return pd.Series(texts)

    def text2vec(self, text):
        cleaned_text = self.cleanup_text(text)
        logger.debug("doc2vec ...")
        vec = [doc.vector for doc in self.nlp.pipe(
            cleaned_text, batch_size=1000, n_threads=4)]
        vec = np.array(vec)
        logger.debug("Vector shape:" + str(vec.shape))
        return vec


# https://www.kaggle.com/opanichev/handcrafted-features/code
class FeatureEnginering():
    def __init__(self, create=True, use_pca=True):
        self.create = create
        self.use_pca = use_pca

    def clean_text(self, x):
        punctuation = ['.', '..', '...', ',',
                       ':', ';', '-', '*', '"', '!', '?']
        x.lower()
        for p in punctuation:
            x.replace(p, '')
        return x

    def extract_features(self, df):
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        # alphabet = 'abcd'
        df['n_.'] = df['text'].str.count('\.')
        df['n_...'] = df['text'].str.count('\...')
        df['n_,'] = df['text'].str.count('\,')
        df['n_:'] = df['text'].str.count('\:')
        df['n_;'] = df['text'].str.count('\;')
        df['n_-'] = df['text'].str.count('\-')
        df['n_?'] = df['text'].str.count('\?')
        df['n_!'] = df['text'].str.count('\!')
        df['n_\''] = df['text'].str.count('\'')
        df['n_"'] = df['text'].str.count('\"')

        # First words in a sentence
        df['n_The '] = df['text'].str.count('The ')
        df['n_I '] = df['text'].str.count('I ')
        df['n_It '] = df['text'].str.count('It ')
        df['n_He '] = df['text'].str.count('He ')
        df['n_Me '] = df['text'].str.count('Me ')
        df['n_She '] = df['text'].str.count('She ')
        df['n_We '] = df['text'].str.count('We ')
        df['n_They '] = df['text'].str.count('They ')
        df['n_You '] = df['text'].str.count('You ')

        is_alphabet = False
        if is_alphabet:
            # Find numbers of different combinations
            for c in tqdm(alphabet.upper()):
                df['n_' + c] = df['text'].str.count(c)
                df['n_' + c + '.'] = df['text'].str.count(c + '\.')
                df['n_' + c + ','] = df['text'].str.count(c + '\,')

                for c2 in alphabet:
                    df['n_' + c + c2] = df['text'].str.count(c + c2)
                    df['n_' + c + c2 +
                        '.'] = df['text'].str.count(c + c2 + '\.')
                    df['n_' + c + c2 +
                        ','] = df['text'].str.count(c + c2 + '\,')

            for c in tqdm(alphabet):
                df['n_' + c + '.'] = df['text'].str.count(c + '\.')
                df['n_' + c + ','] = df['text'].str.count(c + '\,')
                df['n_' + c + '?'] = df['text'].str.count(c + '\?')
                df['n_' + c + ';'] = df['text'].str.count(c + '\;')
                df['n_' + c + ':'] = df['text'].str.count(c + '\:')

                for c2 in alphabet:
                    df['n_' + c + c2 +
                        '.'] = df['text'].str.count(c + c2 + '\.')
                    df['n_' + c + c2 +
                        ','] = df['text'].str.count(c + c2 + '\,')
                    df['n_' + c + c2 +
                        '?'] = df['text'].str.count(c + c2 + '\?')
                    df['n_' + c + c2 +
                        ';'] = df['text'].str.count(c + c2 + '\;')
                    df['n_' + c + c2 +
                        ':'] = df['text'].str.count(c + c2 + '\:')
                    df['n_' + c + ', ' +
                        c2] = df['text'].str.count(c + '\, ' + c2)

            # And now starting processing of cleaned text
            for c in tqdm(alphabet):
                df['n_' + c] = df['text_cleaned'].str.count(c)
                df['n_' + c + ' '] = df['text_cleaned'].str.count(c + ' ')
                df['n_' + ' ' + c] = df['text_cleaned'].str.count(' ' + c)

                for c2 in alphabet:
                    df['n_' + c + c2] = df['text_cleaned'].str.count(c + c2)
                    df['n_' + c + c2 +
                        ' '] = df['text_cleaned'].str.count(c + c2 + ' ')
                    df['n_' + ' ' + c +
                        c2] = df['text_cleaned'].str.count(' ' + c + c2)
                    df['n_' + c + ' ' +
                        c2] = df['text_cleaned'].str.count(c + ' ' + c2)

                    for c3 in alphabet:
                        df['n_' + c + c2 +
                            c3] = df['text_cleaned'].str.count(c + c2 + c3)

        df['n_the'] = df['text_cleaned'].str.count('the ')
        df['n_ a '] = df['text_cleaned'].str.count(' a ')
        df['n_appear'] = df['text_cleaned'].str.count('appear')
        df['n_little'] = df['text_cleaned'].str.count('little')
        df['n_was '] = df['text_cleaned'].str.count('was ')
        df['n_one '] = df['text_cleaned'].str.count('one ')
        df['n_two '] = df['text_cleaned'].str.count('two ')
        df['n_three '] = df['text_cleaned'].str.count('three ')
        df['n_ten '] = df['text_cleaned'].str.count('ten ')
        df['n_is '] = df['text_cleaned'].str.count('is ')
        df['n_are '] = df['text_cleaned'].str.count('are ')
        df['n_ed'] = df['text_cleaned'].str.count('ed ')
        df['n_however'] = df['text_cleaned'].str.count('however')
        df['n_ to '] = df['text_cleaned'].str.count(' to ')
        df['n_into'] = df['text_cleaned'].str.count('into')
        df['n_about '] = df['text_cleaned'].str.count('about ')
        df['n_th'] = df['text_cleaned'].str.count('th')
        df['n_er'] = df['text_cleaned'].str.count('er')
        df['n_ex'] = df['text_cleaned'].str.count('ex')
        df['n_an '] = df['text_cleaned'].str.count('an ')
        df['n_ground'] = df['text_cleaned'].str.count('ground')
        df['n_any'] = df['text_cleaned'].str.count('any')
        df['n_silence'] = df['text_cleaned'].str.count('silence')
        df['n_wall'] = df['text_cleaned'].str.count('wall')
        if "author" in df.columns:
            df.drop(["author"], axis=1, inplace=True)
        return df.drop(['id', 'text', 'text_cleaned'], axis=1)

    # https://www.kaggle.com/phoenix120/lstm-sentence-embeddings-with-additional-features
    def collect_additional_features(self, train, test):
        logger.debug("Collecting additional features")
        train_df = train.copy()
        test_df = test.copy()

        eng_stopwords = set(nltk.corpus.stopwords.words("english"))

        train_df["words"] = train_df["text"].apply(lambda text: text.split())
        test_df["words"] = test_df["text"].apply(lambda text: text.split())

        train_df["num_words"] = train_df["words"].apply(
            lambda words: len(words))
        test_df["num_words"] = test_df["words"].apply(lambda words: len(words))

        train_df["num_unique_words"] = train_df["words"].apply(
            lambda words: len(set(words)))
        test_df["num_unique_words"] = test_df["words"].apply(
            lambda words: len(set(words)))

        train_df["num_chars"] = train_df["text"].apply(lambda text: len(text))
        test_df["num_chars"] = test_df["text"].apply(lambda text: len(text))

        train_df["num_stopwords"] = train_df["words"].apply(
            lambda words: len([w for w in words if w in eng_stopwords]))
        test_df["num_stopwords"] = test_df["words"].apply(
            lambda words: len([w for w in words if w in eng_stopwords]))

        train_df["num_punctuations"] = train_df['text'].apply(
            lambda text: len([c for c in text if c in string.punctuation]))
        test_df["num_punctuations"] = test_df['text'].apply(
            lambda text: len([c for c in text if c in string.punctuation]))

        train_df["num_words_upper"] = train_df["words"].apply(
            lambda words: len([w for w in words if w.isupper()]))
        test_df["num_words_upper"] = test_df["words"].apply(
            lambda words: len([w for w in words if w.isupper()]))

        train_df["num_words_title"] = train_df["words"].apply(
            lambda words: len([w for w in words if w.istitle()]))
        test_df["num_words_title"] = test_df["words"].apply(
            lambda words: len([w for w in words if w.istitle()]))

        train_df["mean_word_len"] = train_df["words"].apply(
            lambda words: np.mean([len(w) for w in words]))
        test_df["mean_word_len"] = test_df["words"].apply(
            lambda words: np.mean([len(w) for w in words]))

        train_df.drop(["text", "id", "words"], axis=1, inplace=True)
        test_df.drop(["text", "id", "words"], axis=1, inplace=True)
        if "author" in train_df.columns:
            train_df.drop(["author"], axis=1, inplace=True)
        if "author" in test_df.columns:
            test_df.drop(["author"], axis=1, inplace=True)
        logger.debug("train_df size:" + str(train_df.shape) +
                     ". test_df size:" + str(test_df.shape))
        return train_df, test_df

    # https://www.kaggle.com/c/spooky-author-identification/discussion/42815
    def determine_analysis_input(self, sentence):
        text = word_tokenize(sentence)
        tagged = pos_tag(text)
        analysis = {}
        analysis["cc"] = len([word for word in tagged if word[1] == "CC"])
        analysis["in"] = len([word for word in tagged if word[1] == "IN"])
        analysis["wh"] = len([word for word in tagged if word[1] == "WP"])
        analysis["wh$"] = len([word for word in tagged if word[1] == "WP$"])
        analysis["md"] = len([word for word in tagged if word[1] == "MD"])
        analysis["present"] = len(
            [word for word in tagged if word[1] in ["VBP", "VBZ", "VBG"]])
        analysis["past"] = len(
            [word for word in tagged if word[1] in ["VBD", "VBN"]])
        analysis['adverb'] = len(
            [word for word in tagged if word[1] in ["RB", "RBR", "RBS"]])
        analysis['adjective'] = len(
            [word for word in tagged if word[1] in ["JJ", "JJR", "JJS"]])
        return analysis

    def extract_feature_pos_tag(self, data):
        # data_out = data.apply(lambda x: self.determine_analysis_input(x))
        data_out = []
        for index, row in data.iterrows():
            row_out = self.determine_analysis_input(row['text'])
            data_out.append(row_out)
        # print(data_out[:5])
        return pd.DataFrame(data_out)

    def process_data(self, train, test):
        train_file = DATA_DIR + "/train_fe.csv"
        test_file = DATA_DIR + "/test_fe.csv"

        if self.create:
            # Extracting features based on statistic
            logger.debug("Extracting features based on statistic method 1")
            train_df, test_df = self.collect_additional_features(train, test)
            logger.debug("Numer of extracted features:" +
                         str(len(train_df.columns)))
            train = train.copy()
            test = test.copy()
            # Extracting additional features
            logger.debug("Extracting features based on pos tag")
            train_df3 = self.extract_feature_pos_tag(train)
            test_df3 = self.extract_feature_pos_tag(test)
            logger.debug("Numer of extracted features:" +
                         str(len(train_df3.columns)))

            logger.debug("Extracting features based statistic method 2")
            train['text_cleaned'] = train['text'].apply(
                lambda x: self.clean_text(x))
            test['text_cleaned'] = test['text'].apply(
                lambda x: self.clean_text(x))
            logger.debug("Extracting feature for train")
            train_df2 = self.extract_features(train)
            logger.debug("Extracting feature for test")
            test_df2 = self.extract_features(test)
            logger.debug("Numer of extracted features:" +
                         str(len(test_df2.columns)))

            # Drop non-relevant columns
            logger.debug('Searching for columns with non-changing values...')
            counts = train_df2.sum(axis=0)
            cols_to_drop = counts[counts == 0].index.values
            train_df2.drop(cols_to_drop, axis=1, inplace=True)
            test_df2.drop(cols_to_drop, axis=1, inplace=True)
            logger.debug('Dropped ' + str(len(cols_to_drop)) + ' columns.')

            logger.debug('Searching for columns with low STD...')
            counts = train_df2.std(axis=0)
            cols_to_drop = counts[counts < 0.01].index.values
            train_df2.drop(cols_to_drop, axis=1, inplace=True)
            test_df2.drop(cols_to_drop, axis=1, inplace=True)
            logger.debug('Dropped ' + str(len(cols_to_drop)) + ' columns.')

            logger.debug("Saving extracted features ...")
            # Combined 3 features set
            train_all = pd.concat([train_df, train_df3, train_df2], axis=1, ignore_index=True)
            test_all = pd.concat([test_df, test_df3, test_df2], axis=1, ignore_index=True)
            # save to file
            train_all.to_csv(train_file, index=False)
            test_all.to_csv(test_file, index=False)
        else:
            logger.debug("Loading extracted features ...")
            train_all = pd.read_csv(train_file)
            test_all = pd.read_csv(test_file)

        logger.debug('train.shape = ' + str(train_all.shape) +
                     ', test.shape = ' + str(test_all.shape))
        logger.debug("Train columns:" + str(train_all.columns))
        # logger.debug("Test columns:" + str(test_all.columns))
        train_all = train_all.values
        test_all = test_all.values
        if self.use_pca:
            logger.debug("PCA to reduce dimension ...")
            pca = decomposition.PCA(n_components=OUTPUT_DIM)
            pca.fit(train_all)
            train_all = pca.transform(train_all)
            test_all = pca.transform(test_all)
        # else:
        #     logger.debug("Scaled transform train and test data")
        #     scaler = MinMaxScaler()
        #     train_all = scaler.fit_transform(train_all)
        #     test_all = scaler.transform(test_all)
        return train_all, test_all


# build extra dataset from corpus:
# https://github.com/llealgt/edgar_philip_love_poe
class BuildExtraDataSet():
    def __init__(self):
        pass

    def process_corpus(self, dir=LOVECRAFT_DIR, code="HPL"):
        logger.debug("Load all stories in " + dir)
        filenames = glob.glob(dir + "/*.txt")
        all_sentenses = []
        for filename in filenames:
            logger.debug("Processing " + filename)
            f = open(filename, "r")
            document = f.read()
            sentences = nltk.sent_tokenize(document)
            for sentence in sentences:
                words = nltk.word_tokenize(sentence)
                if len(words) > 5:
                    all_sentenses.append(sentence.strip())
        logger.debug("Total number of sentenses:" + str(len(all_sentenses)))
        id_list = []
        for i in range(len(all_sentenses)):
            id = id_generator()
            id_list.append(id)

        data = pd.DataFrame({'id': id_list, 'text': all_sentenses})
        data.loc[:, 'author'] = code
        print(data.head(5))
        return data

    def build_dataset(self):
        logger.debug("Building data set from corpus")
        lovecraft = self.process_corpus(dir=LOVECRAFT_DIR, code="HPL")
        poe = self.process_corpus(dir=POE_DIR, code='EAP')
        shelley = self.process_corpus(dir=SHELLEY_DIR, code='MWS')
        train_ex = pd.concat([lovecraft, poe, shelley])
        train_ex.to_csv(DATA_DIR + "/train_ex.csv", index=False)


class SpookyAuthorIdentifer():
    def __init__(self, label, word2vec=0, model_choice=MODEL_FASTEXT, model_choice2=None):
        self.label = label
        self.word2vec = word2vec
        self.model_choice = model_choice
        self.model_choice2 = model_choice2

    def load_data(self):
        logger.info("Loading data ...")
        train_data = pd.read_csv(DATA_DIR + "/train.csv")
        train_data_ex = pd.read_csv(DATA_DIR + "/train_ex.csv")
        self.train_data = pd.concat([train_data, train_data_ex])
        self.eval_data = pd.read_csv(DATA_DIR + "/test.csv")

        logger.debug("train size:" + str(self.train_data.shape) +
                     " test size:" + str(self.eval_data.shape))

    def load_pretrained_word_embedding(self, create=True):
        file = DATA_DIR + "/embeding.pickle"
        if create:
            logger.debug("Loading pre-trained word embbeding ...")
            # load the whole embedding into memory
            embeddings_index = dict()
            f = open(GLOBAL_DATA_DIR + '/glove.6B.300d.txt')
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            f.close()
            logger.debug('Loaded ' + str(len(embeddings_index)))
            logger.debug("Create word matrix")
            # create a weight matrix for words in training docs
            embedding_matrix = np.zeros((self.vocab_size, OUTPUT_DIM))
            for word, i in self.tokenizer.word_index.items():
                embedding_vector = embeddings_index.get(word)
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

    def preprocess_text(self, preprocess=1):
        x_train = self.train_data.text
        x_test = self.eval_data.text
        if preprocess == 1:
            logger.debug("Stemmer text ..")
            # stemmer = stm.SnowballStemmer("english")
            # stemmer = stm.lancaster.LancasterStemmer()
            stemmer = stm.PorterStemmer()
            x_train = x_train.apply(lambda x: (" ").join(
                [stemmer.stem(z) for z in re.sub("[^a-zA-Z0-9]", " ", x).split(" ")]))
            x_test = x_test.apply(lambda x: (" ").join(
                [stemmer.stem(z) for z in re.sub("[^a-zA-Z0-9]", " ", x).split(" ")]))
            x_all = pd.concat([x_train, x_test])
        elif preprocess == 2:
            logger.debug("Lemmtizer text ..")
            alpha_tokenizer = RegexpTokenizer('[A-Za-z]\w+')
            lemmatizer = WordNetLemmatizer()
            stop = stopwords.words('english')
            x_train_list = [[lemmatizer.lemmatize(word.lower()) for word in alpha_tokenizer.tokenize(
                sent) if word.lower() not in stop] for sent in x_train]
            x_test_list = [[lemmatizer.lemmatize(word.lower()) for word in alpha_tokenizer.tokenize(
                sent) if word.lower() not in stop] for sent in x_test]
            x_train = [" ".join(x) for x in x_train_list]
            x_test = [" ".join(x) for x in x_test_list]
            x_all = x_train + x_test
        else:
            x_all = pd.concat([x_train, x_test])
        logger.debug("X_all size:" + str(len(x_all)))
        print(x_train[0])
        return x_all, x_train, x_test

    def tokenize(self, x_all, x_train, x_test):
        logger.debug("Tokenizing text ..")
        # prepare tokenizer
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(x_all)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        logger.debug("vocab size:" + str(self.vocab_size))
        # integer encode the documents
        logger.debug("Text to sequences")
        x_train_tokenized = self.tokenizer.texts_to_sequences(x_train)
        x_test_tokenized = self.tokenizer.texts_to_sequences(x_test)
        logger.debug("X train shape:" + str(len(x_train_tokenized)))
        print(x_train_tokenized[0])
        if USE_SEQUENCE is False:
            logger.debug("Text to maxtrix")
            x_train_matrix = self.tokenizer.texts_to_matrix(
                x_train, mode='tfidf')
            x_test_matrix = self.tokenizer.texts_to_matrix(
                x_test, mode='tfidf')
            logger.debug("X train matrix shape:" + str((x_train_matrix.shape)))
            logger.debug("X test matrix shape:" + str((x_test_matrix.shape)))
        else:
            x_train_matrix = None
            x_test_matrix = None
        return x_train_tokenized, x_test_tokenized, x_train_matrix, x_test_matrix

    def build_word2vec(self, data, create=True):
        file = w2v_weight_path
        if create:
            logger.debug("Generating word2vec ....")
            w2v = Word2Vec(data, min_count=1, size=OUTPUT_DIM, workers=4)
            # create a weight matrix for words in training docs
            embedding_matrix = np.zeros((self.vocab_size, OUTPUT_DIM))
            for word, i in self.tokenizer.word_index.items():
                try:
                    embedding_vector = w2v.wv[word]
                    embedding_matrix[i] = embedding_vector
                except:
                    pass
            with open(file, 'wb') as f:
                # Pickle the 'data' dictionary using the highest protocol
                # available.
                pickle.dump(embedding_matrix, f, pickle.HIGHEST_PROTOCOL)
        else:
            logger.debug("Loading from pickle")
            with open(file, 'rb') as f:
                embedding_matrix = pickle.load(f)
        self.w2v_weights = embedding_matrix
        logger.debug("w2v size:" + str(self.w2v_weights.shape))

    def tokenize_and_ngram(self):
        x_all, x_train, x_test = self.preprocess_text(preprocess=1)
        x_train, x_test, x_train_matrix, x_test_matrix = self.tokenize(
            x_all, x_train, x_test)
        # self.tokenize(x_all, x_train, x_test)
        if USE_SEQUENCE:
            # Create n-gram
            if ngram_range > 1:
                logger.debug("Creating n-gram ...:" + str(ngram_range))
                # Create set of unique n-gram from the training set.
                ngram_set = set()
                for input_list in x_train:
                    for i in range(2, ngram_range + 1):
                        set_of_ngram = self.create_ngram_set(
                            input_list, ngram_value=i)
                        ngram_set.update(set_of_ngram)

                # Dictionary mapping n-gram token to a unique integer.
                # Integer values are greater than max_features in order
                # to avoid collision with existing features.
                start_index = self.vocab_size + 1
                token_indice = {v: k + start_index for k,
                                v in enumerate(ngram_set)}
                indice_token = {token_indice[k]: k for k in token_indice}

                # max_features is the highest integer that could be found in the
                # dataset.
                self.vocab_size = np.max(list(indice_token.keys())) + 1
                logger.debug("New vocab size:" + str(self.vocab_size))

                # Augmenting x_train and x_test with n-grams features
                x_train = self.add_ngram(x_train, token_indice, ngram_range)
                x_test = self.add_ngram(x_test, token_indice, ngram_range)
                print(x_train[0])

            max_train_sequence = np.amax(list(map(len, x_train)))
            max_test_sequence = np.amax(list(map(len, x_test)))
            print('Average train sequence length: {}'.format(
                np.mean(list(map(len, x_train)), dtype=int)))
            logger.debug('Max train sequence length: ' +
                         str(max_train_sequence))
            print('Average test sequence length: {}'.format(
                np.mean(list(map(len, x_test)), dtype=int)))
            logger.debug('Max test sequence length: ' + str(max_test_sequence))

            self.input_length = max(
                max_train_sequence, max_test_sequence, SEQUENCE_LENGTH)
            logger.debug('Old sequence length:' + str(SEQUENCE_LENGTH) +
                         '. New squence length:' + str(self.input_length))

            # pad documents to a max length
            logger.debug("Sequence padding ...")
            self.train = pad_sequences(
                x_train, maxlen=self.input_length)
            self.X_eval = pad_sequences(
                x_test, maxlen=self.input_length)
            logger.debug("Train shape after padding:" + str(self.train.shape))
            # print(self.train[0])
        else:
            self.train = x_train_matrix
            self.X_eval = x_test_matrix
            self.input_length = self.vocab_size

        if (self.word2vec == 1):
            self.build_word2vec(x_all, False)

    def prepare_data(self):
        if self.model_choice2 == MODEL_INPUT2_DENSE or self.model_choice == MODEL_INPUT2_DENSE:
            logger.debug("Fature engineering")
            fe = FeatureEnginering(create=True, use_pca=False)
            self.train_df, self.eval_df = fe.process_data(
                self.train_data, self.eval_data)
        # CREATE TARGET VARIABLE
        self.eval_id = self.eval_data['id']
        logger.debug("One hot encoding for label")
        self.train_data["EAP"] = (self.train_data.author == "EAP") * 1
        self.train_data["HPL"] = (self.train_data.author == "HPL") * 1
        self.train_data["MWS"] = (self.train_data.author == "MWS") * 1
        self.target_vars = ["EAP", "HPL", "MWS"]
        self.target = self.train_data[self.target_vars].values
        if USE_SPACY:
            spacy_object = SpacyPreprocess()
            self.train = spacy_object.text2vec(self.train_data.text)
            self.X_eval = spacy_object.text2vec(self.eval_data.text)
            self.vocab_size = self.train.shape[1]
            self.input_length = self.vocab_size
            logger.debug("Vocab size:" + str(self.vocab_size))
        else:
            self.tokenize_and_ngram()

    def buil_embbeding_layer(self):
        logger.debug("Input length:" + str(self.input_length))
        if self.word2vec == 2:
            # load pre-trained word embedding
            embedding_matrix = self.load_pretrained_word_embedding(True)
            embedding_layer = Embedding(self.vocab_size, OUTPUT_DIM, weights=[
                embedding_matrix], input_length=self.input_length, trainable=True)
        elif self.word2vec == 1:
            # Use gensim generated word2vec
            embedding_layer = Embedding(self.vocab_size, OUTPUT_DIM, weights=[
                self.w2v_weights], input_length=self.input_length, trainable=True)
        else:
            embedding_layer = Embedding(
                self.vocab_size, OUTPUT_DIM, input_length=self.input_length, trainable=True)

        return embedding_layer

    def model_fasttext(self):
        embbeding_input = Input(shape=(None,), name="embbeding_input")
        embedding_layer = self.buil_embbeding_layer()(embbeding_input)
        model = Dropout(dropout, seed=random_state)(embedding_layer)
        model = GlobalAveragePooling1D()(model)
        model = Dropout(dropout, seed=random_state)(model)
        return embbeding_input, model

    def model_cnn(self):
        if KERAS_EMBEDDING:
            input1 = Input(shape=(None,), name="embbeding_input_cnn")
            embedding_layer = self.buil_embbeding_layer()(input1)
            model = Dropout(dropout, seed=random_state)(embedding_layer)
        else:
            input1 = Reshape((1, self.input_length),
                             input_shape=(self.input_length,))
            model = input1
        for i in range(KERAS_LAYERS):
            model = Conv1D(OUTPUT_DIM, KERAS_KERNEL_SIZE,
                           activation='relu')(model)
            model = MaxPooling1D(pool_size=KERAS_POOL_SIZE)(model)
            model = Dropout(dropout, seed=random_state)(model)
        model = GlobalMaxPooling1D()(model)
        model = Dropout(dropout, seed=random_state)(model)
        model = BatchNormalization()(model)
        return input1, model

    # https://richliao.github.io/supervised/classification/2016/11/26/textclassifier-convolutional/
    # https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras/blob/master/sentiment_cnn.py
    def model_cnn2(self):

        embbeding_input = Input(shape=(None,), name="embbeding_input")
        embedding_layer = self.buil_embbeding_layer()(embbeding_input)
        model = Dropout(dropout, seed=random_state)(embedding_layer)
        filter_sizes = [3, 4, 5]
        conv_blocks = []
        for fsz in filter_sizes:
            conv = Conv1D(filters=KERAS_FILTERS,
                          kernel_size=fsz,
                          padding="valid",
                          activation="relu",
                          strides=1)(model)
            conv = MaxPooling1D(pool_size=KERAS_POOL_SIZE)(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)
        model_merge = concatenate(conv_blocks)
        model = Dropout(dropout, seed=random_state)(model_merge)
        return embbeding_input, model

    def model_cudnnlstm(self):
        embbeding_input = Input(shape=(None,), name="embbeding_input_lstm")
        embedding_layer = self.buil_embbeding_layer()(embbeding_input)
        model = Dropout(dropout, seed=random_state)(embedding_layer)
        # for i in range(KERAS_LAYERS):
        #     model = Conv1D(OUTPUT_DIM, KERAS_KERNEL_SIZE,
        #                    activation='relu')(model)
        #     model = MaxPooling1D(pool_size=KERAS_POOL_SIZE)(model)
        #     model = Dropout(dropout, seed=random_state)(model)
        # LSTM
        model = CuDNNLSTM(OUTPUT_DIM)(model)
        model = Dropout(dropout, seed=random_state)(model)
        model = BatchNormalization()(model)
        return embbeding_input, model

    def model_lstm(self):
        embbeding_input = Input(shape=(None,), name="embbeding_input_lstm")
        embedding_layer = self.buil_embbeding_layer()(embbeding_input)
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

    # https://richliao.github.io/supervised/classification/2016/12/26/textclassifier-RNN/
    # https://github.com/synthesio/hierarchical-attention-networks/blob/master/model.py
    def model_lstm_attrnn(self):
        embbeding_input = Input(
            shape=(self.input_length,), name="embbeding_input_lstm")
        embedding_layer = self.buil_embbeding_layer()(embbeding_input)
        model = Dropout(dropout, seed=random_state)(embedding_layer)
        # LSTM
        model = Bidirectional(
            GRU(OUTPUT_DIM // 2, return_sequences=True))(model)
        # model = GRU(OUTPUT_DIM, return_sequences=True)(model)
        model = Dropout(dropout, seed=random_state)(model)
        model = AttentionWithContext()(model)
        model = Dropout(dropout, seed=random_state)(model)
        return embbeding_input, model

    # https://github.com/tqtg/textClassifier/blob/master/rnn_classififer.py
    def model_lstm_attrnn2(self):
        sentence_input = Input(shape=(self.input_length,), dtype='int32')
        embedding_layer = self.buil_embbeding_layer()
        embedded_sequences = embedding_layer(sentence_input)
        l_dropout1 = Dropout(KERAS_DROPOUT_RATE)(embedded_sequences)

        # ================================ Bidirectional LSTM model ===========
        # l_lstm = Bidirectional(LSTM(OUTPUT_DIM//2))(l_dropout1)
        # l_dropout2 = Dropout(dropout)(l_lstm)
        # l_classifier = Dense(2, activation='softmax')(l_dropout2)
        # ================================ One-level attention RNN (GRU) ======
        h_word = Bidirectional(
            GRU(OUTPUT_DIM // 2, return_sequences=True), name='h_word')(l_dropout1)
        # h_word = AttentionWithContext()(h_word)
        h_word = Dropout(KERAS_DROPOUT_RATE)(h_word)
        # Attention part
        u_word = TimeDistributed(
            Dense(OUTPUT_DIM, activation='tanh'), name='u_word')(h_word)
        u_word = Dropout(KERAS_DROPOUT_RATE)(u_word)
        # \alpha weight for each word
        alpha_word = TimeDistributed(Dense(1, use_bias=False))(u_word)
        alpha_word = Reshape((self.input_length,))(alpha_word)
        alpha_word = Activation('softmax')(alpha_word)
        alpha_word = Dropout(KERAS_DROPOUT_RATE)(alpha_word)
        # Combine word representation to form sentence representation w.r.t
        # \alpha weights
        h_word_combined = Dot(axes=[1, 1])([h_word, alpha_word])
        h_word_combined = Dropout(KERAS_DROPOUT_RATE)(h_word_combined)
        # l_classifier = Dense(2, activation='softmax', name='classifier')(h_word_combined)
        return sentence_input, h_word_combined

    # https://richliao.github.io/supervised/classification/2016/12/26/textclassifier-HATN/
    def model_lstm_he_attrnn_old(self):
        embbeding_input = Input(
            shape=(self.input_length,), name="embbeding_input_lstm")
        embedding_layer = self.buil_embbeding_layer()(embbeding_input)
        model = Dropout(dropout, seed=random_state)(embedding_layer)
        model = Bidirectional(
            GRU(OUTPUT_DIM, return_sequences=True))(model)
        model = TimeDistributed(Dense(OUTPUT_DIM))(model)
        model = AttentionWithContext()(model)
        model = Dropout(dropout, seed=random_state)(model)
        sentEncoder = Model(embbeding_input, model)

        review_input = Input(shape=(None, self.input_length))
        review_encoder = TimeDistributed(sentEncoder)(review_input)
        l_lstm_sent = Bidirectional(
            GRU(OUTPUT_DIM, return_sequences=True))(review_encoder)
        l_dense_sent = TimeDistributed(Dense(OUTPUT_DIM))(l_lstm_sent)
        l_att_sent = AttentionWithContext()(l_dense_sent)
        att_model = Dropout(dropout, seed=random_state)(l_att_sent)
        # model = Model(review_input, att_model)
        return embbeding_input, att_model

    # https://github.com/tqtg/textClassifier/blob/master/hatt_classifier.py
    def model_lstm_he_attrnn(self):
        # Word level
        sentence_input = Input(shape=(self.input_length,), dtype='int32')
        embedding_layer = self.buil_embbeding_layer()
        embedded_sequences = embedding_layer(sentence_input)
        l_dropout1 = Dropout(dropout)(embedded_sequences)

        h_word = Bidirectional(
            GRU(OUTPUT_DIM, return_sequences=True), name='h_word')(l_dropout1)
        u_word = TimeDistributed(
            Dense(2 * OUTPUT_DIM, activation='tanh'), name='u_word')(h_word)

        alpha_word = TimeDistributed(Dense(1, use_bias=False))(u_word)
        alpha_word = Reshape((self.input_length,))(alpha_word)
        alpha_word = Activation('softmax')(alpha_word)

        h_word_combined = Dot(axes=[1, 1], name='h_word_combined')(
            [h_word, alpha_word])

        sent_encoder = Model(sentence_input, h_word_combined)
        sent_encoder.summary()

        # Sentence level
        review_input = Input(
            shape=(OUTPUT_DIM, self.input_length), dtype='int32')
        review_encoder = TimeDistributed(
            sent_encoder, name='sent_encoder')(review_input2)
        l_dropout2 = Dropout(dropout)(review_encoder)

        h_sent = Bidirectional(
            GRU(OUTPUT_DIM, return_sequences=True), name='h_sent')(l_dropout2)
        u_sent = TimeDistributed(
            Dense(2 * OUTPUT_DIM, activation='tanh'), name='u_sent')(h_sent)

        alpha_sent = TimeDistributed(Dense(1, use_bias=False))(u_sent)
        alpha_sent = Reshape((OUTPUT_DIM,))(alpha_sent)
        alpha_sent = Activation('softmax')(alpha_sent)

        h_sent_combined = Dot(axes=[1, 1], name='h_sent_combined')(
            [h_sent, alpha_sent])

        # Classifier layer
        # l_classifier = Dense(5, activation='softmax')(h_sent_combined)
        return review_input, h_sent_combined

    def model_input2_dense(self):
        n_features = self.train_df.shape[1]
        logger.debug("Num features of input2:" + str(n_features))
        feature_input = Input(shape=(n_features,), name="features_input")
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
            if nodes < 16:
                nodes = 16
        logger.debug("Input2 model shape:" + str(model.shape))
        return feature_input, model

    def model_input2_cnn(self):
        n_features = self.train_df.shape[1]
        logger.debug("Num features of input2:" + str(n_features))
        feature_input = Input(shape=(n_features,), name="features_input")
        model = Reshape((KERAS_KERNEL_SIZE, n_features //
                         KERAS_KERNEL_SIZE))(feature_input)
        # nodes = max(n_features, OUTPUT_DIM)
        nodes = min(n_features * 2, OUTPUT_DIM)
        layers = max(KERAS_LAYERS, 1)
        for i in range(layers):
            model = Conv1D(nodes, KERAS_KERNEL_SIZE,
                           activation='relu')(model)
            model = MaxPooling1D(pool_size=1)(model)
            model = Dropout(KERAS_DROPOUT_RATE, seed=random_state)(model)
        model = GlobalMaxPooling1D()(model)
        model = Dropout(KERAS_DROPOUT_RATE, seed=random_state)(model)
        # model = (Dense(OUTPUT_DIM, activation='relu',
        #                kernel_constraint=keras.constraints.maxnorm(KERAS_MAXNORM)))(model)
        # model = Dropout(KERAS_DROPOUT_RATE, seed=random_state)(model)
        model = BatchNormalization()(model)
        logger.debug("Input2 model shape:" + str(model.shape))
        return feature_input, model

    def build_model(self):
        logger.debug("Model definition")
        if self.model_choice == MODEL_FASTEXT:
            input1, model1 = self.model_fasttext()
        elif self.model_choice == MODEL_CUDNNLSTM:
            input1, model1 = self.model_cudnnlstm()
        elif self.model_choice == MODEL_LSTM:
            input1, model1 = self.model_lstm()
        elif self.model_choice == MODEL_CNN:
            # input1, model1 = self.model_cnn()
            input1, model1 = self.model_cnn2()
        elif self.model_choice == MODEL_LSTM_ATTRNN:
            input1, model1 = self.model_lstm_attrnn2()
        elif self.model_choice == MODEL_LSTM_HE_ATTRNN:
            input1, model1 = self.model_lstm_he_attrnn()

        elif self.model_choice == MODEL_INPUT2_DENSE:
            input1, model1 = self.model_input2_dense()
            # input1, model1 = self.model_input2_cnn()
            # Change train set
            self.train = self.train_df
            self.X_eval = self.eval_df

        if self.model_choice2 is not None:  # Use additional input2
            logger.debug("Adding more model...")
            if self.model_choice2 == MODEL_INPUT2_DENSE:
                input2, model2 = self.model_input2_dense()
                # input2, model2 = self.model_input2_cnn()
            elif self.model_choice2 == MODEL_FASTEXT:
                input2, model2 = self.model_fasttext()
            elif self.model_choice2 == MODEL_CUDNNLSTM:
                input2, model2 = self.model_cudnnlstm()
            elif self.model_choice2 == MODEL_LSTM:
                input2, model2 = self.model_lstm()
            elif self.model_choice2 == MODEL_CNN:
                # input2, model2 = self.model_cnn()
                input2, model2 = self.model_cnn2()
            elif self.model_choice2 == MODEL_LSTM_ATTRNN:
                input2, model2 = self.model_lstm_attrnn2()

            model3 = concatenate([model1, model2])
            n_features = int(model3.shape[1])
            logger.debug("Concatenate feature size:" + str(n_features))
            # model3 = Average()([model1, model2])
            # out_model = Reshape((2, n_features//2))(model3)
            # out_model = GlobalAveragePooling1D()(out_model)
            out_model = Dropout(KERAS_DROPOUT_RATE,
                                seed=random_state)(model3)
            # out_model = Dense(3, activation='relu', kernel_constraint=keras.constraints.maxnorm(
            #     KERAS_MAXNORM))(out_model)
            # out_model = Dropout(KERAS_DROPOUT_RATE,
            #                     seed=random_state)(out_model)
            # out_model = BatchNormalization()(out_model)
            out_model = Dense(3, activation='softmax')(out_model)
            self.model = Model(inputs=[input1, input2], outputs=out_model)
        else:
            out_model = Dense(3, activation='softmax')(model1)
            self.model = Model(inputs=[input1], outputs=out_model)
        # compile the model
        optimizer = Adam(lr=KERAS_LEARNING_RATE, decay=decay)
        # optimizer = RMSprop(lr=KERAS_LEARNING_RATE, decay=decay)
        self.model.compile(optimizer=optimizer,
                           loss='binary_crossentropy', metrics=['acc'])
        # summarize the model
        print(self.model.summary())
        # Plot the model
        plot_model(self.model, show_shapes=True,
                   to_file=DATA_DIR + '/model.png')

    def train_single_model(self):
        self.build_model()
        # Use Early-Stopping
        callback_early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=KERAS_EARLY_STOPPING, verbose=VERBOSE, mode='auto')
        callback_tensorboard = keras.callbacks.TensorBoard(
            log_dir=DATA_DIR + '/tensorboard', histogram_freq=1, batch_size=KERAS_BATCH_SIZE,
            write_graph=True, write_grads=True, write_images=False)
        callback_checkpoint = keras.callbacks.ModelCheckpoint(
            model_weight_path, monitor='val_loss', verbose=VERBOSE, save_best_only=True, mode='auto')
        logger.debug(" Spliting train and test set...")

        if self.model_choice2 == MODEL_INPUT2_DENSE:
            X_train, X_test, X_train2, X_test2, Y_train, Y_test = train_test_split(
                self.train, self.train_df, self.target, test_size=KERAS_VALIDATION_SPLIT, random_state=1234)
            logger.debug("X_train:" + str(X_train.shape) +
                         ". X_test:" + str(X_test.shape))
            logger.debug("X_train2:" + str(X_train2.shape) +
                         ". X_test2:" + str(X_test2.shape))
        else:
            X_train, X_test, Y_train, Y_test = train_test_split(
                self.train, self.target, test_size=KERAS_VALIDATION_SPLIT, random_state=1234)
            logger.debug("X_train:" + str(X_train.shape) +
                         ". X_test:" + str(X_test.shape))

        logger.debug("Training ...")
        start = time.time()
        # Training model
        if self.model_choice2 is not None:
            if self.model_choice2 != MODEL_INPUT2_DENSE:
                X_train2 = X_train
                X_test2 = X_test
            self.history = self.model.fit([X_train, X_train2], Y_train,
                                          validation_data=(
                [X_test, X_test2], Y_test),
                batch_size=KERAS_BATCH_SIZE,
                epochs=KERAS_N_ROUNDS,
                callbacks=[
                # callback_tensorboard,
                callback_early_stopping,
                callback_checkpoint,
            ],
                verbose=VERBOSE
            )

        else:
            self.history = self.model.fit(X_train, Y_train,
                                          validation_data=(X_test, Y_test),
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
        self.model.load_weights(model_weight_path)
        logger.debug('Best metric:' + str(callback_early_stopping.best))
        logger.debug(
            'Best round:' + str(callback_early_stopping.stopped_epoch - KERAS_EARLY_STOPPING))

    def train_kfold_single(self):
        logger.info("Train Kfold for single model")
        self.build_model()
        # Save initilize model weigth for reseting weigth after each loop
        model_init_weights = self.model.get_weights()

        logger.debug("Prepare training data ...")
        X = self.train
        Y = self.target
        T = self.X_eval

        if self.model_choice2 is not None:
            if self.model_choice2 == MODEL_INPUT2_DENSE:
                X2 = self.train_df
                T2 = self.eval_df
            else:
                X2 = X
                T2 = T

        # S_train = np.zeros((X.shape[0], N_FOLDS))
        Y_eval = np.zeros((T.shape[0], Y.shape[1], N_FOLDS))
        total_time = 0
        total_metric = 0
        total_best_round = 0
        self.history_total = []

        kfolds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=321)
        for j, (train_idx, test_idx) in enumerate(kfolds.split(X)):
            logger.debug("Round:" + str(j + 1))
            start = time.time()
            X_train = X[train_idx]
            Y_train = Y[train_idx]
            X_test = X[test_idx]
            Y_test = Y[test_idx]

            if self.model_choice2 is not None:
                X_train2 = X2[train_idx]
                X_test2 = X2[test_idx]

            # Use Early-Stopping
            callback_early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=KERAS_EARLY_STOPPING, verbose=VERBOSE, mode='auto')
            callback_tensorboard = keras.callbacks.TensorBoard(
                log_dir=DATA_DIR + '/tensorboard', histogram_freq=1, batch_size=KERAS_BATCH_SIZE,
                write_graph=True, write_grads=True, write_images=False)
            callback_checkpoint = keras.callbacks.ModelCheckpoint(
                model_weight_path, monitor='val_loss', verbose=VERBOSE, save_best_only=True, mode='auto')

            # Training model
            if self.model_choice2 is not None:
                history = self.model.fit([X_train, X_train2], Y_train,
                                         validation_data=(
                                             [X_test, X_test2], Y_test),
                                         batch_size=KERAS_BATCH_SIZE,
                                         epochs=KERAS_N_ROUNDS,
                                         callbacks=[
                    # callback_tensorboard,
                    callback_early_stopping,
                    callback_checkpoint,
                ],
                    verbose=VERBOSE
                )
            else:
                history = self.model.fit(X_train, Y_train,
                                         validation_data=(X_test, Y_test),
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
            total_time = total_time + end
            self.history_total.append(history)
            # load best model
            logger.info("Loading best model ...")
            self.model.load_weights(model_weight_path)
            metric = callback_early_stopping.best
            total_metric = total_metric + metric
            logger.debug('Best metric:' + str(metric))
            best_round = callback_early_stopping.stopped_epoch - KERAS_EARLY_STOPPING
            total_best_round = total_best_round + best_round
            logger.debug('Best round:' + str(best_round))
            logger.debug("Saving Y_eval for round:" + str(j + 1))
            if self.model_choice2 is not None:
                Y_eval[:, :, j] = self.model.predict([T, T2])
            else:
                Y_eval[:, :, j] = self.model.predict(T)
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
            if self.model_choice2 is not None:
                if self.model_choice2 == MODEL_INPUT2_DENSE:
                    Y_pred = self.model.predict([self.X_eval, self.eval_df])
                else:
                    Y_pred = self.model.predict([self.X_eval, self.X_eval])
            else:
                Y_pred = self.model.predict(self.X_eval)
        preds = pd.DataFrame(Y_pred, columns=self.target_vars)
        eval_output = pd.concat([self.eval_id, preds], 1)
        today = str(dtime.date.today())
        logger.debug("Date:" + today)
        eval_output.to_csv(
            DATA_DIR + '/' + today + '-submission.csv.gz', index=False, float_format='%.5f',
            compression='gzip')

    def plot_history(self):
        history = self.history
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

    def plot_kfold_history(self):
        history = self.history_total
        plt.figure(figsize=(20, 15))
        plt.subplot(2, 1, 1)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        for i, history in enumerate(self.history_total):
            plt.plot(history.history['loss'], label='train_loss' + str(i + 1))
            plt.plot(history.history['val_loss'],
                     label='test_loss' + str(i + 1))
            # plt.legend(['train', 'test'], loc='upper left')
        plt.legend()
        # Summmary auc score
        plt.subplot(2, 1, 2)
        plt.title('model acc')
        plt.ylabel('acc')
        plt.xlabel('epoch')
        for i, history in enumerate(self.history_total):
            plt.plot(history.history['acc'], label='train_acc' + str(i + 1))
            plt.plot(history.history['val_acc'],
                     label='test_acc' + str(i + 1))
        plt.legend()
        # plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(DATA_DIR + "/history.png")


# ---------------- Main -------------------------
if __name__ == "__main__":
    pd.options.display.float_format = '{:,.5f}'.format
    logger = logging.getLogger('spooky-author-identification')
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

    # set GPU memory
    # KTF.set_session(set_gpu_memory())
    label = 'author'
    object = SpookyAuthorIdentifer(
        label, word2vec=2, model_choice=MODEL_INPUT2_DENSE, model_choice2=None)
    option = 1
    if option == 0:
        data_obj = BuildExtraDataSet()
        data_obj.build_dataset()
    elif option == 1:
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

    logger.info("Done!")
