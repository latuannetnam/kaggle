# Toxic Comment Classification Challenge
# Identify and classify toxic online comments
# https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
# Credit:

# References
# https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
# https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
# https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
# https://martinbel.github.io/fast-text.html
# http://debajyotidatta.github.io/nlp/deep/learning/word-embeddings/2016/09/28/fast-text-and-skip-gram/
# http://ben.bolte.cc/blog/2016/gensim.html
# https://www.bonaccorso.eu/2017/08/07/twitter-sentiment-analysis-with-gensim-word2vec-and-keras-convolutional-networks/
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


def id_generator(size=6, chars=string.digits):
    return 'id' + ''.join(rn.choice(chars) for _ in range(size))


def id_generator2(size=7):
    id_generator2.counter += 1
    return 'id' + str(id_generator2.counter).zfill(size)


# https://www.kaggle.com/fmuetsch/keras-nn-with-rec-layers-sentiment-etc-2
def _get_stem_single(args):
    data, index = args
    logger.debug("Stemmer text ..")
    # stemmer = stm.SnowballStemmer("english")
    # stemmer = stm.lancaster.LancasterStemmer()
    stemmer = stm.PorterStemmer()
    data_stem = data.apply(lambda x: (" ").join(
        [stemmer.stem(z) for z in re.sub("[^a-zA-Z0-9]", " ", x).split(" ")]))
    return pd.Series(data_stem, index=index)


def _tokenize_single(args):
    key, texts = args
    logger.debug("Tokenizing text for " + key)
    # prepare tokenizer
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(texts)
    vocab_size = len(tokenizer.word_index) + 1
    logger.debug(key + ":vocab size:" + str(vocab_size))
    vocab_size = min(vocab_size, VOCAB_SIZE)
    # integer encode the documents
    logger.debug(key + ":Text to sequences")
    new_texts = tokenizer.texts_to_sequences(texts)
    max_sequence_length = np.amax(list(map(len, new_texts)))
    logger.debug(key + ":Sequence length:" + str(max_sequence_length))
    # pad documents to a max length
    new_texts = pad_sequences(new_texts, max_sequence_length)
    logger.debug(key + ":Squence shape:" + str(new_texts.shape))
    return key, vocab_size, max_sequence_length, new_texts


def keras_rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


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
    def __init__(self, create=True, label='comment_text', use_pca=False):
        self.create = create
        self.label = label
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
    def collect_additional_features(self, texts):
        logger.debug("Collecting additional features")
        texts_df = texts.copy()
        label = self.label
        print(texts_df[:3])
        eng_stopwords = set(nltk.corpus.stopwords.words("english"))

        texts_df["words"] = texts_df[label].apply(lambda text: text.split())
        texts_df["num_words"] = texts_df["words"].apply(
            lambda words: len(words))
        texts_df["num_unique_words"] = texts_df["words"].apply(
            lambda words: len(set(words)))
        # texts_df["num_unique_words"] = texts_df["words"].apply(
        #     lambda words: self.unique_words(words))
        texts_df["num_chars"] = texts_df[label].apply(lambda text: len(text))
        texts_df["num_stopwords"] = texts_df["words"].apply(
            lambda words: len([w for w in words if w in eng_stopwords]))
        texts_df["num_punctuations"] = texts_df[label].apply(
            lambda text: len([c for c in text if c in string.punctuation]))
        texts_df["num_words_upper"] = texts_df["words"].apply(
            lambda words: len([w for w in words if w.isupper()]))
        texts_df["num_words_title"] = texts_df["words"].apply(
            lambda words: len([w for w in words if w.istitle()]))
        texts_df["mean_word_len"] = texts_df["words"].apply(
            lambda words: np.mean([len(w) for w in words]))
        texts_df.drop([label, 'words'], axis=1, inplace=True)
        return texts_df

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
        data_out = []
        for index, row in data.iterrows():
            row_out = self.determine_analysis_input(row[self.label])
            data_out.append(row_out)
        return pd.DataFrame(data_out)

    def process_data(self, texts):
        train_file = DATA_DIR + "/train_fe.csv"
        test_file = DATA_DIR + "/test_fe.csv"

        if self.create:
            # Extracting features based on statistic
            logger.debug("Extracting features based on statistic method 1")
            train = texts.loc['train'].astype(str).copy()
            test = texts.loc['eval'].astype(str).copy()

            train_df = self.collect_additional_features(train)
            test_df = self.collect_additional_features(test)

            logger.debug("Numer of extracted features:%d",
                         len(train_df.columns))
            # Extracting additional features
            logger.debug("Extracting features based on pos tag")
            train_df2 = self.extract_feature_pos_tag(train)
            test_df2 = self.extract_feature_pos_tag(test)
            logger.debug("Numer of extracted features:%d",
                         len(train_df2.columns))
            # Combined 3 features set
            train_all = pd.concat([train_df, train_df2], axis=1)
            test_all = pd.concat([test_df, test_df2], axis=1)
            # Drop non-relevant columns
            logger.debug('Searching for columns with non-changing values...')
            counts = train_all.sum(axis=0)
            cols_to_drop = counts[counts == 0].index.values
            train_all.drop(cols_to_drop, axis=1, inplace=True)
            test_all.drop(cols_to_drop, axis=1, inplace=True)
            logger.debug('Dropped ' + str(len(cols_to_drop)) + ' columns.')

            logger.debug('Searching for columns with low STD...')
            counts = train_all.std(axis=0)
            cols_to_drop = counts[counts < 0.01].index.values
            train_all.drop(cols_to_drop, axis=1, inplace=True)
            test_all.drop(cols_to_drop, axis=1, inplace=True)
            logger.debug('Dropped ' + str(len(cols_to_drop)) + ' columns.')
            # save to file
            train_all.to_csv(train_file, index=False)
            test_all.to_csv(test_file, index=False)
        else:
            logger.debug("Loading extracted features ...")
            train_all = pd.read_csv(train_file)
            test_all = pd.read_csv(test_file)
        train_all.fillna(value=0, inplace=True)
        test_all.fillna(value=0, inplace=True)
        logger.debug('train.shape = ' + str(train_all.shape) +
                     ', test.shape = ' + str(test_all.shape))
        logger.debug("Train columns:" + str(train_all.columns))
        return train_all, test_all


class ToxicCommentClassification():
    def __init__(self, label, word2vec=0, model_choice=MODEL_CNN):
        self.label = label
        self.word2vec = word2vec
        self.model_choice = model_choice
        self.vocab_size = OUTPUT_DIM
        self.embedding_features = ['comment_text']
        # self.embedding_features = ['name', 'item_description']
        self.special_features = None

    def get_embedding_features(self):
        return self.embedding_features

    def load_data(self):
        logger.info("Loading data ...")
        # train_data = pd.read_csv(DATA_DIR + "/train.csv", nrows=100)
        # eval_data = pd.read_csv(DATA_DIR + "/test.csv", nrows=100)
        train_data = pd.read_csv(DATA_DIR + "/train.csv")
        eval_data = pd.read_csv(DATA_DIR + "/test.csv")

        self.combine_data = pd.concat(
            [train_data, eval_data], keys=['train', 'eval'])
        logger.debug("train size:" + str(train_data.shape) +
                     " test size:" + str(eval_data.shape))

    def load_pretrained_word_embedding(self, tokenizer, key, create=True):
        logger.debug("Build embbeding matrix from pre-trained word2vec Glove")
        file = OUTPUT_DIR + "/embeding-" + key + ".pickle"
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
            # matrix_size = len(tokenizer.word_index) + 1
            embedding_matrix = np.zeros((self.vocab_size[key], OUTPUT_DIM))

            for word, i in tokenizer.word_index.items():
                if i >= self.vocab_size[key]:
                    continue
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

    def build_word2vec(self, data, create=True):
        file = w2v_weight_path
        if create:
            logger.debug("Generating word2vec ....")
            w2v = Word2Vec(data, min_count=1, size=OUTPUT_DIM, workers=4)
            # create a weight matrix for words in training docs
            matrix_size = len(self.tokenizer.word_index) + 1
            embedding_matrix = np.zeros((matrix_size, OUTPUT_DIM))
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
        logger.debug("w2v size:" + str(self.embedding_matrix.shape))
        return embedding_matrix

    def get_stem(self, data, index):
        p = Pool(processes=N_THREADS)
        n = math.ceil(len(data) / N_THREADS)
        stems = p.map(_get_stem_single, [
            (data[i:i + n], index[i:i + n]) for i in range(0, len(data), n)])
        # return np.array(flatten(stems))
        # return stems
        return pd.concat(stems)

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

        logger.debug("Done Preprocessing ")
        return new_texts

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

    def tokenize_and_ngram(self):
        self.vocab_size = {}
        self.sequence_length = {}

        for key in self.embedding_features:
            logger.debug("Tokenizing for " + key)
            tokenizer = Tokenizer(num_words=VOCAB_SIZE)
            tokenizer.fit_on_texts(self.combine_data[key])
            vocab_size = len(tokenizer.word_index) + 1
            logger.debug("vocab size:" + str(vocab_size))
            self.vocab_size[key] = min(vocab_size, VOCAB_SIZE)
            # integer encode the documents
            logger.debug("Text to sequences")
            self.combine_data[key] = tokenizer.texts_to_sequences(
                self.combine_data[key])
            # Create n-gram
            if NGRAM_RANGE > 1:
                logger.debug("Creating n-gram ...:" + str(NGRAM_RANGE))
                # Create set of unique n-gram from the training set.
                ngram_set = set()
                for input_list in self.combine_data[key]:
                    for i in range(2, NGRAM_RANGE + 1):
                        set_of_ngram = self.create_ngram_set(
                            input_list, ngram_value=i)
                        ngram_set.update(set_of_ngram)

                # Dictionary mapping n-gram token to a unique integer.
                # Integer values are greater than max_features in order
                # to avoid collision with existing features.
                start_index = self.vocab_size[key] + 1
                token_indice = {v: k + start_index for k,
                                v in enumerate(ngram_set)}
                indice_token = {token_indice[k]: k for k in token_indice}

                # max_features is the highest integer that could be found in the
                # dataset.
                self.vocab_size[key] = np.max(list(indice_token.keys())) + 1
                logger.debug("New vocab size:" + str(self.vocab_size[key]))

                # Augmenting x_train and x_test with n-grams features
                self.combine_data[key] = self.add_ngram(
                    self.combine_data[key], token_indice, NGRAM_RANGE)

            max_sequence_length = np.amax(
                list(map(len, self.combine_data[key])))
            self.sequence_length[key] = min(
                max_sequence_length, SEQUENCE_LENGTH)
            logger.debug("Max sequence length:%d. Sequence length:%d",
                         max_sequence_length, self.sequence_length[key])
            # pad documents to a max length
            self.combine_data[key] = pad_sequences(
                self.combine_data[key], self.sequence_length[key])
            logger.debug(key + ":Train shape after padding:" +
                         str(self.combine_data[key].shape))
            logger.debug("Vocabulary size:" + str(self.vocab_size[key]))

            if self.word2vec == 1:
                self.embedding_matrix = self.load_pretrained_word_embedding(
                    tokenizer, key, create=False)

    def handle_missing(self, dataset):
        dataset.comment_text.fillna(value="missing", inplace=True)
        return (dataset)

    def prepare_data(self):

        logger.debug("Data features preparing...")
        # Handle missing value
        self.combine_data = self.handle_missing(self.combine_data)
        for key in self.embedding_features:
            self.combine_data[key] = self.preprocess_text(
                self.combine_data[key], preprocess=3)
        # feature engineering
        extra_object = FeatureEnginering(
            create=False, label='comment_text', use_pca=False)
        train_df, test_df = extra_object.process_data(
            self.combine_data[self.embedding_features])
        del extra_object
        self.numeric_features = train_df.columns.values
        # Tokenize text data
        self.tokenize_and_ngram()
        # Split train and test set
        logger.debug("Split train and eval data set")
        self.train_data = pd.concat(
            [self.combine_data.loc['train'], train_df], axis=1)
        self.eval_data = pd.concat(
            [self.combine_data.loc['eval'], test_df], axis=1)
        del self.combine_data

        # CREATE TARGET VARIABLE
        self.list_classes = ["toxic", "severe_toxic",
                             "obscene", "threat", "insult", "identity_hate"]
        self.target = self.train_data[self.list_classes].values
        self.eval_id = self.eval_data['id']
        logger.debug("train size:" + str(self.train_data.shape))
        logger.debug("test size:" + str(self.eval_data.shape))

    def get_keras_data(self, dataset):
        X = {}
        for key in self.embedding_features:
            X[key] = dataset[key].values
        X['num_vars'] = dataset[self.numeric_features].values
        return X

    def buil_embbeding_layer(self, vocab_size, output_dim, input_length, word2vec=0):
        if word2vec > 0:
            embedding_layer = Embedding(vocab_size, output_dim, weights=[
                self.embedding_matrix], input_length=input_length, trainable=True)
        else:
            embedding_layer = Embedding(
                vocab_size, output_dim, input_length=input_length, trainable=True)
        return embedding_layer

    def model_fasttext(self, name, vocab_size, output_dim, input_length, word2vec=0):
        logger.debug("Building FastText model ...")
        embbeding_input = Input(shape=(None,), name=name)
        embedding_layer = self.buil_embbeding_layer(
            vocab_size, output_dim, input_length, word2vec)(embbeding_input)
        model = Dropout(dropout, seed=random_state)(embedding_layer)
        model = GlobalAveragePooling1D()(model)
        model = Dropout(dropout, seed=random_state)(model)
        return embbeding_input, model

    def model_cnn(self, name, vocab_size, output_dim, input_length, word2vec=0):
        logger.debug("Building CNN model ...")
        embbeding_input = Input(shape=(None,), name=name)
        embedding_layer = self.buil_embbeding_layer(
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
        logger.debug("Building CNN2 model ...")
        embbeding_input = Input(shape=(None,), name=name)
        embedding_layer = self.buil_embbeding_layer(
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
    def model_cnn3(self, input_length):
        logger.debug("Building CNN3 model ...")
        embbeding_input = Input(shape=(None,), name=name)
        embedding_layer = self.buil_embbeding_layer(
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

    def model_cudnnlstm(self, name, vocab_size, output_dim, input_length, word2vec=0):
        logger.debug("Building CuDNN LSTM model ...")
        embbeding_input = Input(shape=(None,), name=name)
        embedding_layer = self.buil_embbeding_layer(
            vocab_size, output_dim, input_length, word2vec)(embbeding_input)
        model = Dropout(dropout, seed=random_state)(embedding_layer)
        # LSTM
        model = CuDNNLSTM(output_dim)(model)
        model = Dropout(dropout, seed=random_state)(model)
        # model = BatchNormalization()(model)
        return embbeding_input, model

    def model_cudnnlstm2(self, name, vocab_size, output_dim, input_length, word2vec=0):
        logger.debug("Building Bidirectional CuDNN LSTM model ...")
        embbeding_input = Input(shape=(None,), name=name)
        embedding_layer = self.buil_embbeding_layer(
            vocab_size, output_dim, input_length, word2vec)(embbeding_input)
        model = Dropout(dropout, seed=random_state)(embedding_layer)
        # LSTM
        model = Bidirectional(
            CuDNNLSTM(output_dim // 2, return_sequences=True))(model)
        model = GlobalMaxPooling1D()(model)
        model = Dropout(dropout, seed=random_state)(model)
        # model = BatchNormalization()(model)
        return embbeding_input, model

    def model_lstm(self, input_length):
        logger.debug("Building LSTM model ...")
        embbeding_input = Input(shape=(None,))
        embedding_layer = self.buil_embbeding_layer(
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

    # https://richliao.github.io/supervised/classification/2016/12/26/textclassifier-RNN/
    # https://github.com/synthesio/hierarchical-attention-networks/blob/master/model.py
    def model_lstm_attrnn(self, name, vocab_size, output_dim, input_length, word2vec=0):
        logger.debug("Building Attention model ...")
        embbeding_input = Input(shape=(None,), name=name)
        embedding_layer = self.buil_embbeding_layer(
            vocab_size, output_dim, input_length, word2vec)(embbeding_input)
        model = Dropout(dropout, seed=random_state)(embedding_layer)
        # LSTM
        model = Bidirectional(
            CuDNNGRU(output_dim // 2, return_sequences=True))(model)
        # model = GRU(OUTPUT_DIM, return_sequences=True)(model)
        model = Dropout(dropout, seed=random_state)(model)
        model = AttentionWithContext()(model)
        model = Dropout(dropout, seed=random_state)(model)
        return embbeding_input, model

    # https://github.com/tqtg/textClassifier/blob/master/rnn_classififer.py
    def model_lstm_attrnn2(self, name, vocab_size, output_dim, input_length, word2vec=0):
        logger.debug("Building Attention2 model ...")
        embbeding_input = Input(shape=(None,), name=name)
        embedding_layer = self.buil_embbeding_layer(
            vocab_size, output_dim, input_length, word2vec)(embbeding_input)
        l_dropout1 = Dropout(dropout)(embedding_layer)

        # ================================ Bidirectional LSTM model ===========
        # l_lstm = Bidirectional(LSTM(OUTPUT_DIM//2))(l_dropout1)
        # l_dropout2 = Dropout(dropout)(l_lstm)
        # l_classifier = Dense(2, activation='softmax')(l_dropout2)
        # ================================ One-level attention RNN (GRU) ======
        # h_word = Bidirectional(
        # GRU(OUTPUT_DIM // 2, return_sequences=True),
        # name='h_word')(l_dropout1)
        h_word = Bidirectional(
            CuDNNGRU(output_dim // 2, return_sequences=True))(l_dropout1)
        # h_word = AttentionWithContext()(h_word)
        h_word = Dropout(dropout)(h_word)
        # Attention part
        u_word = TimeDistributed(
            Dense(output_dim, activation='tanh'))(h_word)
        u_word = Dropout(dropout)(u_word)
        # \alpha weight for each word
        alpha_word = TimeDistributed(Dense(1, use_bias=False))(u_word)
        alpha_word = Reshape((input_length,))(alpha_word)
        alpha_word = Activation('softmax')(alpha_word)
        alpha_word = Dropout(dropout)(alpha_word)
        # Combine word representation to form sentence representation w.r.t
        # \alpha weights
        h_word_combined = Dot(axes=[1, 1])([h_word, alpha_word])
        h_word_combined = Dropout(dropout)(h_word_combined)
        return embbeding_input, h_word_combined

    # https://richliao.github.io/supervised/classification/2016/12/26/textclassifier-HATN/
    def model_lstm_he_attrnn_old(self, input_length):
        logger.debug("Building Attention Hierachical model ...")
        embbeding_input = Input(
            shape=(self.input_length,))
        embedding_layer = self.buil_embbeding_layer()(embbeding_input)
        model = Dropout(dropout, seed=random_state)(embedding_layer)
        # model = Bidirectional(
        #     GRU(OUTPUT_DIM // 2, return_sequences=True))(model)
        model = Bidirectional(
            CuDNNGRU(OUTPUT_DIM // 2, return_sequences=True))(model)
        model = TimeDistributed(Dense(OUTPUT_DIM))(model)
        model = AttentionWithContext()(model)
        model = Dropout(dropout, seed=random_state)(model)
        sentEncoder = Model(embbeding_input, model)

        review_input = Input(shape=(None, self.input_length))
        review_encoder = TimeDistributed(sentEncoder)(review_input)
        # l_lstm_sent = Bidirectional(
        #     GRU(OUTPUT_DIM // 2, return_sequences=True))(review_encoder)
        l_lstm_sent = Bidirectional(
            CuDNNGRU(OUTPUT_DIM // 2, return_sequences=True))(review_encoder)
        l_dense_sent = TimeDistributed(Dense(OUTPUT_DIM))(l_lstm_sent)
        l_att_sent = AttentionWithContext()(l_dense_sent)
        att_model = Dropout(dropout, seed=random_state)(l_att_sent)
        # model = Model(review_input, att_model)
        return embbeding_input, att_model

    # https://github.com/tqtg/textClassifier/blob/master/hatt_classifier.py
    def model_lstm_he_attrnn(self, input_length):
        # Word level
        sentence_input = Input(shape=(self.input_length,), dtype='int32')
        embedding_layer = self.buil_embbeding_layer(input_length)
        embedded_sequences = embedding_layer(sentence_input)
        l_dropout1 = Dropout(dropout)(embedded_sequences)

        # h_word = Bidirectional(
        # GRU(OUTPUT_DIM // 2, return_sequences=True),
        # name='h_word')(l_dropout1)
        h_word = Bidirectional(
            CuDNNGRU(OUTPUT_DIM // 2, return_sequences=True))(l_dropout1)
        u_word = TimeDistributed(
            Dense(OUTPUT_DIM, activation='tanh'))(h_word)

        alpha_word = TimeDistributed(Dense(1, use_bias=False))(u_word)
        alpha_word = Reshape((self.input_length,))(alpha_word)
        alpha_word = Activation('softmax')(alpha_word)

        h_word_combined = Dot(axes=[1, 1])(
            [h_word, alpha_word])

        sent_encoder = Model(sentence_input, h_word_combined)
        sent_encoder.summary()

        # Sentence level
        review_input = Input(
            shape=(OUTPUT_DIM, self.input_length), dtype='int32')
        review_encoder = TimeDistributed(
            sent_encoder)(review_input)
        l_dropout2 = Dropout(dropout)(review_encoder)

        h_sent = Bidirectional(
            GRU(OUTPUT_DIM, return_sequences=True))(l_dropout2)
        u_sent = TimeDistributed(
            Dense(2 * OUTPUT_DIM, activation='tanh'))(h_sent)

        alpha_sent = TimeDistributed(Dense(1, use_bias=False))(u_sent)
        alpha_sent = Reshape((OUTPUT_DIM,))(alpha_sent)
        alpha_sent = Activation('softmax')(alpha_sent)

        h_sent_combined = Dot(axes=[1, 1])(
            [h_sent, alpha_sent])

        # Classifier layer
        # l_classifier = Dense(5, activation='softmax')(h_sent_combined)
        return review_input, h_sent_combined

    def model_input2_dense(self):
        logger.debug("Building dense model from sencondary input ...")
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

    def build_model(self, X_train):
        logger.debug("Model definition")
        key_var = 'num_vars'
        if key_var in X_train:
            vocab_size = int(np.max(X_train[key_var]) + 1)
            sequence_lenght = X_train[key_var].shape[1]
            logger.debug("Vocab size for num_vars:" + str(vocab_size) +
                         " . Sequence length:" + str(sequence_lenght))
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

        elif self.model_choice == MODEL_LSTM_ATTRNN:
            # model = self.model_lstm_attrnn2
            model = self.model_lstm_attrnn

        elif self.model_choice == MODEL_LSTM_HE_ATTRNN:
            model = self.model_lstm_he_attrnn

        # Build models for each features in dataset
        inputs = []
        in_models = []
        for key in self.embedding_features:
            input, in_model = model(
                key, self.vocab_size[key], OUTPUT_DIM, self.sequence_length[key], word2vec=1)
            inputs.append(input)
            in_models.append(in_model)
        if key_var in X_train:
            input, in_model = model(
                key_var, vocab_size, sequence_lenght, sequence_lenght, word2vec=0)
            inputs.append(input)
            in_models.append(in_model)

        if len(in_models) > 1:
            model_all = concatenate(in_models)
            model_all = Dropout(KERAS_DROPOUT_RATE,
                                seed=random_state)(model_all)
        else:
            model_all = in_models[0]

        out_model = Dense(len(self.list_classes),
                          activation="sigmoid")(model_all)
        self.model = Model(
            inputs=inputs, outputs=out_model)
        # self.num_inputs = len(inputs)
        # compile the model
        optimizer = Adam(lr=KERAS_LEARNING_RATE, decay=decay)
        # optimizer = RMSprop(lr=KERAS_LEARNING_RATE, decay=decay)
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy')
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

    def train_single_model(self):
        logger.info("Training for single model ...")
        logger.debug(" Spliting train and test set...")
        self.build_model(self.get_keras_data(self.train_data))
        d_train, d_valid, Y_train, Y_test = train_test_split(
            self.train_data, self.target, test_size=KERAS_VALIDATION_SPLIT, shuffle=False, random_state=1234)
        X_train = self.get_keras_data(d_train)
        X_test = self.get_keras_data(d_valid)

        # Use Early-Stopping
        callback_early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=KERAS_EARLY_STOPPING, verbose=VERBOSE, mode='auto')
        callback_tensorboard = keras.callbacks.TensorBoard(
            log_dir=OUTPUT_DIR + '/tensorboard', histogram_freq=1, batch_size=KERAS_BATCH_SIZE,
            write_graph=True, write_grads=True, write_images=False)
        callback_checkpoint = keras.callbacks.ModelCheckpoint(
            model_weight_path, monitor='val_loss', verbose=VERBOSE, save_best_only=True, mode='auto')

        logger.debug("Training ...")
        start = time.time()
        # Training model
        self.history = self.model.fit([X_train[key] for key in X_train], Y_train,
                                      validation_data=(
            [X_test[key] for key in X_test], Y_test),
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
        X_comment_text = X['comment_text']
        Y = self.target
        T = self.get_keras_data(self.eval_data)
        T_comment_text = T['comment_text']
        # S_train = np.zeros((X.shape[0], N_FOLDS))
        Y_eval = np.zeros((T_comment_text.shape[0], Y.shape[1], N_FOLDS))
        logger.debug("Y eval size:" + str(Y_eval.shape))
        total_time = 0
        total_metric = 0
        total_best_round = 0
        self.history_total = []
        # build model and train
        self.build_model(X)
        # Save initilize model weigth for reseting weigth after each loop
        model_init_weights = self.model.get_weights()
        # Setup KFold
        kfolds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=321)
        for j, (train_idx, test_idx) in enumerate(kfolds.split(X_comment_text)):
            logger.debug("Round:" + str(j + 1))
            start = time.time()
            X_train = {}
            X_test = {}
            # Split train and test data
            for key in X:
                X_train[key] = X[key][train_idx]
                X_test[key] = X[key][test_idx]
                logger.debug(
                    key + ":Train size:" + str(X_train[key].shape) + " .Test size:" + str(X_test[key].shape))

            Y_train = Y[train_idx]
            Y_test = Y[test_idx]

            # Use Early-Stopping
            callback_early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=KERAS_EARLY_STOPPING, verbose=VERBOSE, mode='auto')
            callback_tensorboard = keras.callbacks.TensorBoard(
                log_dir=OUTPUT_DIR + '/tensorboard', histogram_freq=1, batch_size=KERAS_BATCH_SIZE,
                write_graph=True, write_grads=True, write_images=False)
            callback_checkpoint = keras.callbacks.ModelCheckpoint(
                model_weight_path, monitor='val_loss', verbose=VERBOSE, save_best_only=True, mode='auto')
            # Training
            history = self.model.fit([X_train[key] for key in X_train], Y_train,
                                     validation_data=(
                [X_test[key] for key in X_test], Y_test),
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
            best_round = callback_early_stopping.stopped_epoch + 1 - KERAS_EARLY_STOPPING
            total_best_round = total_best_round + best_round
            logger.debug('Best round:' + str(best_round))
            # Predict
            logger.debug("Saving Y_eval for round:" + str(j + 1))
            Y_eval[:, :, j] = self.model.predict([T[key] for key in T])
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
            Y_pred = self.model.predict([X_eval[key] for key in X_eval])
        preds = pd.DataFrame(Y_pred, columns=[self.list_classes])
        eval_output = pd.concat([self.eval_id, preds], 1)
        today = str(dtime.date.today())
        logger.debug("Date:" + today)
        eval_output.to_csv(
            DATA_DIR + '/' + today + '-submission.csv.gz', index=False, float_format='%.5f',
            compression='gzip')

    def plot_history(self):
        history = self.history
        plt.figure(figsize=(20, 15))
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
        # plt.subplot(2, 1, 1)
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

    # Global variables
    manager = Manager()

    label = 'comment_text'
    object = ToxicCommentClassification(
        label, word2vec=1, model_choice=MODEL_CUDNNLSTM)

    option = 2
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
    elif option == 10:
        object.load_data()
        object.prepare_data()
        object.load_model()
        object.predict_data()

    end = time.time() - start
    logger.info("Total time:" + str(end))
    logger.info("Done!")
