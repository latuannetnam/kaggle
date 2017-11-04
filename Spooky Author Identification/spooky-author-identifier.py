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

# NLP libraries
import nltk
import nltk.stem as stm
import re
from nltk.tokenize import RegexpTokenizer
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import spacy
import string

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


# Keras
import keras
from keras.models import Sequential, Model
from keras.layers import concatenate, Input, Dense, Dropout, BatchNormalization, Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D, LSTM, CuDNNLSTM, Bidirectional, GlobalAveragePooling1D, Reshape
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import SGD, Adam, RMSprop
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
SEQUENCE_LENGTH = 500
OUTPUT_DIM = 500
KERAS_LEARNING_RATE = 0.001
KERAS_N_ROUNDS = 100
KERAS_BATCH_SIZE = 64
KERAS_NODES = 1024
KERAS_LAYERS = 0
KERAS_DROPOUT_RATE = 0.5
# KERAS_REGULARIZER = KERAS_LEARNING_RATE/10
KERAS_REGULARIZER = 0
KERAS_VALIDATION_SPLIT = 0.2
KERAS_EARLY_STOPPING = 5
KERAS_MAXNORM = 3
KERAS_PREDICT_BATCH_SIZE = 1024
# ConvNet
KERAS_FILTERS = OUTPUT_DIM
KERAS_KERNEL_SIZE = 2
KERAS_POOL_SIZE = 2

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
ngram_range = 1

# Model choice
MODEL_FASTEXT = 1
MODEL_CUDNNLSTM = 2
MODEL_LSTM = 3
MODEL_CNN = 4

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


class SpookyAuthorIdentifer():
    def __init__(self, label, word2vec=0, use_input2=False, model_choice=MODEL_FASTEXT):
        self.label = label
        self.word2vec = word2vec
        self.use_input2 = use_input2
        self.model_choice = model_choice

    def load_data(self):
        logger.info("Loading data ...")
        self.train_data = pd.read_csv(DATA_DIR + "/train.csv")
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
            logger.debug("Train shape:" + str(self.train.shape))
            # print(self.train[0])
        else:
            self.train = x_train_matrix
            self.X_eval = x_test_matrix
            self.input_length = self.vocab_size

        if (self.word2vec == 1):
            self.build_word2vec(x_all, True)

    # https://www.kaggle.com/phoenix120/lstm-sentence-embeddings-with-additional-features
    def collect_additional_features(self):
        logger.debug("Collecting additional features")
        train_df = self.train_data.copy()
        test_df = self.eval_data.copy()

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
        scaler = MinMaxScaler()
        self.train_df = scaler.fit_transform(train_df)
        self.eval_df = scaler.transform(test_df)
        logger.debug("train_df size:" + str(self.train_df.shape) +
                     ". test_df size:" + str(self.eval_df.shape))

    def prepare_data(self):
        if self.use_input2:
            self.collect_additional_features()
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
                embedding_matrix], input_length=self.input_length, trainable=False)
        elif self.word2vec == 1:
            # Use gensim generated word2vec
            embedding_layer = Embedding(self.vocab_size, OUTPUT_DIM, weights=[
                self.w2v_weights], input_length=self.input_length, trainable=False)
        else:
            embedding_layer = Embedding(
                self.vocab_size, OUTPUT_DIM, input_length=self.input_length)

        return embedding_layer

    def model_fasttext_old(self):
        model = Sequential()
        embedding_layer = self.buil_embbeding_layer()
        model.add(embedding_layer)
        model.add(Dropout(dropout, seed=random_state))
        for i in range(KERAS_LAYERS):
            model.add(Conv1D(OUTPUT_DIM, KERAS_KERNEL_SIZE, activation='relu'))
            model.add(MaxPooling1D(pool_size=KERAS_POOL_SIZE))
            model.add(Dropout(dropout, seed=random_state))
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(dropout, seed=random_state))
        # model.add(BatchNormalization())
        # model.add(Dense(OUTPUT_DIM, activation='relu'))
        # model.add(Dropout(dropout, seed=random_state))
        # model.add(BatchNormalization())
        # model.add(Dense(3, activation='softmax'))
        return model

    def model_fasttext(self):
        embbeding_input = Input(shape=(None,), name="embbeding_input")
        embedding_layer = self.buil_embbeding_layer()(embbeding_input)

        model = Dropout(dropout, seed=random_state)(embedding_layer)
        for i in range(KERAS_LAYERS):
            model = Conv1D(OUTPUT_DIM, KERAS_KERNEL_SIZE,
                           activation='relu')(model)
            model = MaxPooling1D(pool_size=KERAS_POOL_SIZE)(model)
            model = Dropout(dropout, seed=random_state)(model)
        model = GlobalAveragePooling1D()(model)
        model = Dropout(dropout, seed=random_state)(model)
        return embbeding_input, model

    def model_cnn(self):
        model = Sequential()
        if KERAS_EMBEDDING:
            embedding_layer = self.buil_embbeding_layer()
            model.add(embedding_layer)
            model.add(Dropout(dropout, seed=random_state))
        else:
            model.add(Reshape((1, self.input_length),
                              input_shape=(self.input_length,)))
        for i in range(KERAS_LAYERS):
            model.add(Conv1D(OUTPUT_DIM, KERAS_KERNEL_SIZE, activation='relu'))
            model.add(MaxPooling1D(pool_size=KERAS_POOL_SIZE))
            model.add(Dropout(dropout, seed=random_state))
        model.add(GlobalMaxPooling1D())
        model.add(Dropout(dropout, seed=random_state))
        # model.add(BatchNormalization())
        model.add(Dense(OUTPUT_DIM, activation='relu'))
        model.add(Dropout(dropout, seed=random_state))
        model.add(BatchNormalization())
        # model.add(Dense(3, activation='softmax'))
        return model

    def model_cudnnlstm(self):
        model = Sequential()
        embedding_layer = self.buil_embbeding_layer()
        model.add(embedding_layer)
        model.add(Dropout(dropout, seed=random_state))
        for i in range(KERAS_LAYERS):
            model.add(Conv1D(OUTPUT_DIM, KERAS_KERNEL_SIZE, activation='relu'))
            model.add(MaxPooling1D(pool_size=KERAS_POOL_SIZE))
            model.add(Dropout(dropout, seed=random_state))
        # LSTM
        model.add(CuDNNLSTM(OUTPUT_DIM))
        # lstm = LSTM(OUTPUT_DIM, activation='relu', dropout=dropout, recurrent_dropout=dropout)
        # model.add(Bidirectional(lstm))
        model.add(Dropout(dropout, seed=random_state))
        model.add(BatchNormalization())
        # model.add(Dense(3, activation='softmax'))
        return model

    def model_lstm(self):
        model = Sequential()
        embedding_layer = self.buil_embbeding_layer()
        model.add(embedding_layer)
        model.add(Dropout(dropout, seed=random_state))
        for i in range(KERAS_LAYERS):
            model.add(Conv1D(OUTPUT_DIM, KERAS_KERNEL_SIZE, activation='relu'))
            model.add(MaxPooling1D(pool_size=KERAS_POOL_SIZE))
            model.add(Dropout(dropout, seed=random_state))
        # LSTM
        # model.add(CuDNNLSTM(OUTPUT_DIM))
        lstm = LSTM(OUTPUT_DIM, activation='relu',
                    dropout=dropout, recurrent_dropout=dropout)
        # model.add(Bidirectional(lstm))
        model.add(lstm)
        # model.add(Dropout(dropout, seed=random_state))
        model.add(BatchNormalization())
        # model.add(Dense(3, activation='softmax'))
        return model

    def model_input2_dense(self):
        n_features = self.train_df.shape[1]
        logger.debug("Num features of input2:" + str(n_features))
        feature_input = Input(shape=(n_features,), name="features_input")
        model = Dropout(KERAS_DROPOUT_RATE, seed=random_state)(feature_input)
        nodes = OUTPUT_DIM
        layers = max(KERAS_LAYERS, 1)
        for i in range(layers):
            model = (Dense(nodes,
                           activation='relu', kernel_constraint=keras.constraints.maxnorm(KERAS_MAXNORM)))(model)
            model = Dropout(KERAS_DROPOUT_RATE, seed=random_state)(model)
            model = BatchNormalization()(model)
            nodes = int(nodes // 2)
            if nodes < 32:
                nodes = 32
        return feature_input, model

    def train_single_model(self):
        logger.debug("Model definition")
        # Use additional input2
        if self.use_input2:
            input1, model1 = self.model_fasttext_combined()
            input2, model2 = self.model_input2_dense()
            model3 = concatenate([model1, model2])
            out_model = Dense(3, activation='softmax')(model3)
            self.model = Model(inputs=[input1, input2], outputs=out_model)
        else:
            if self.model_choice == MODEL_FASTEXT:
                self.model = self.model_fasttext()
            elif self.model_choice == MODEL_CUDNNLSTM:
                self.model = self.model_cudnnlstm()
            elif self.model_choice == MODEL_LSTM:
                self.model = self.model_lstm()
            elif self.model_choice == MODEL_CNN:
                self.model = self.model_cnn()
            self.model.add(Dense(3, activation='softmax'))

        # compile the model
        optimizer = Adam(lr=KERAS_LEARNING_RATE, decay=decay)
        # optimizer = RMSprop(lr=KERAS_LEARNING_RATE, decay=decay)
        self.model.compile(optimizer=optimizer,
                           loss='binary_crossentropy', metrics=['acc'])
        # summarize the model
        print(self.model.summary())
        # Use Early-Stopping
        callback_early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=KERAS_EARLY_STOPPING, verbose=VERBOSE, mode='auto')
        callback_tensorboard = keras.callbacks.TensorBoard(
            log_dir=DATA_DIR + '/tensorboard', histogram_freq=1, batch_size=KERAS_BATCH_SIZE,
            write_graph=True, write_grads=True, write_images=False)
        callback_checkpoint = keras.callbacks.ModelCheckpoint(
            model_weight_path, monitor='val_loss', verbose=VERBOSE, save_best_only=True, mode='auto')
        logger.debug(" Spliting train and test set...")

        if self.use_input2:
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
        if self.use_input2:
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
        kfolds = KFold(n_splits=N_FOLDS, shuffle=True, random_state=321)
        # compile the model
        optimizer = Adam(lr=KERAS_LEARNING_RATE, decay=decay)

        X = self.train
        Y = self.target
        T = self.X_eval
        S_train = np.zeros((X.shape[0], N_FOLDS))
        Y_eval = np.zeros((T.shape[0], Y.shape[1], N_FOLDS))
        total_time = 0
        total_metric = 0
        self.history_total = []
        logger.debug("Model definition")
        # keras.backend.clear_session()
        # tf.reset_default_graph()
        if self.model_choice == MODEL_FASTEXT:
            self.model = self.model_fasttext()
        elif self.model_choice == MODEL_CUDNNLSTM:
            self.model = self.model_cudnnlstm()
        elif self.model_choice == MODEL_LSTM:
            self.model = self.model_lstm()
        elif self.model_choice == MODEL_CNN:
            self.model = self.model_cnn()
        self.model.add(Dense(3, activation='softmax'))

        self.model.compile(optimizer=optimizer,
                           loss='binary_crossentropy', metrics=['acc'])
        # summarize the model
        print(self.model.summary())
        # Save initilize model weigth for reseting weigth after each loop
        model_init_weights = self.model.get_weights()

        for j, (train_idx, test_idx) in enumerate(kfolds.split(X)):
            logger.debug("Round:" + str(j + 1))
            start = time.time()
            X_train = X[train_idx]
            Y_train = Y[train_idx]
            X_test = X[test_idx]
            Y_test = Y[test_idx]
            # Use Early-Stopping
            callback_early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=KERAS_EARLY_STOPPING, verbose=VERBOSE, mode='auto')
            callback_tensorboard = keras.callbacks.TensorBoard(
                log_dir=DATA_DIR + '/tensorboard', histogram_freq=1, batch_size=KERAS_BATCH_SIZE,
                write_graph=True, write_grads=True, write_images=False)
            callback_checkpoint = keras.callbacks.ModelCheckpoint(
                model_weight_path, monitor='val_loss', verbose=VERBOSE, save_best_only=True, mode='auto')

            # Training model
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
            logger.debug(
                'Best round:' + str(callback_early_stopping.stopped_epoch - KERAS_EARLY_STOPPING))
            logger.debug("Saving Y_eval for round:" + str(j + 1))
            Y_eval[:, :, j] = self.model.predict(T)
            # Reset weigth
            logger.debug("Reset model weights")
            self.model.set_weights(model_init_weights)

            logger.debug("Done training for round:" + str(j + 1) +
                         " time:" + str(end) + "/" + str(total_time))
        logger.debug("Avg metric:" + str(total_metric / (j + 1)))
        Y_eval_total = Y_eval.mean(2)
        logger.debug("Y eval size:" + str(Y_eval_total.shape))
        logger.debug("Total training time:" + str(total_time))
        return Y_eval_total

    def predict_data(self, Y_pred=None):
        # PREDICTION
        logger.debug("Prediction")
        if Y_pred is None:
            if self.use_input2:
                Y_pred = self.model.predict([self.X_eval, self.eval_df])
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
        label, word2vec=0, use_input2=True, model_choice=MODEL_FASTEXT)
    object.load_data()
    object.prepare_data()
    option = 1
    if option == 1:
        object.train_single_model()
        object.predict_data()
        object.plot_history()
    elif option == 2:
        Y_pred = object.train_kfold_single()
        object.predict_data(Y_pred)
        object.plot_kfold_history()

    logger.info("Done!")
