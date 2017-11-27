# Spooky Author Identification
# Share code and discuss insights to identify horror authors from their writings
# https://www.kaggle.com/c/spooky-author-identification
# Credit:
# https://www.kaggle.com/dex314/spooky-lgb-pipeline-sklearn-features
import pandas as pd
# ------------------ Math libraries
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
from scipy.stats import norm
from math import sqrt
from scipy import stats
from scipy import sparse
# ------------------- SKlearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score, train_test_split, learning_curve, validation_curve, KFold
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_decision_regions, plot_learning_curves, plot_confusion_matrix

# XGBoost
from xgboost.sklearn import XGBClassifier

# LightGBM
from lightgbm import LGBMClassifier
import lightgbm as lgb

# --------------- NLP ----------------
# NLTK
import nltk
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import ToktokTokenizer
# spaCy
from spacy.en import English
# gensim
from gensim.models.word2vec import Word2Vec
from collections import Counter, defaultdict
# --------- System ---------
import datetime as dtime
import datetime
import sys
from inspect import getsourcefile
import os.path
import re
import time
import string
import pickle
import warnings
import logging
# warnings.filterwarnings('ignore')
# %matplotlib inline
# Input data files are available in the DATA_DIR directory.
DATA_DIR = "data-temp"
LABEL = 'author'
TEXT_COL = 'text'
# A custom stoplist
STOPLIST = set(sw.words('english') +
               ["n't", "'s", "'m", "ca"] + list(ENGLISH_STOP_WORDS))
# List of symbols we don't care about
SYMBOLS = " ".join(string.punctuation).split(
    " ") + ["-----", "---", "...", "“", "”", "'ve"]

# Maximum number of features for text extraction
MAX_FEATURES = 8000
# MAX_FEATURES = 16000
# Reduce feature dimension
LSA_FEATURES = 2000

# Maximum number of thread to train word2vec model
W2V_N_THREADS = 4

# type of feature extraction
TF_IDF = 1
TF_IDF_NLTK = 2
TF_IDF_W2C = 3
TF_IDF_SPACY = 4

# Learning parameter
N_ROUNDS = 5000
N_FOLDS = 5
EARLY_STOPPING = 5
LEARNING_RATE = 0.01


class SpookyAuthorIdentifer():
    def __init__(self, label, tf_idf_type, use_lsa=False):
        self.label = label
        self.tf_idf_type = tf_idf_type
        self.use_lsa = use_lsa
        self.tf_filename = DATA_DIR + "/tfidf_" + \
            str(self.tf_idf_type) + ".npz"
        # initialize stemmer for NLTK processing
        self.stemmer = PorterStemmer()
        # initilize spaCy parse for English language
        # self.parser = English()

    def load_data(self):
        # Load data.
        logger.info("Loading data ...")
        train_data = pd.read_csv(DATA_DIR + "/train.csv")
        train_data_ex = pd.read_csv(DATA_DIR + "/train_ex.csv")
        self.train_data = pd.concat(
            [train_data, train_data_ex], ignore_index=True)
        self.eval_data = pd.read_csv(DATA_DIR + "/test.csv")
        self. combine_data = pd.concat(
            [self.train_data, self.eval_data], keys=['train', 'eval'])

        logger.debug("train size:" + str(self.train_data.shape) +
                     " test size:" + str(self.eval_data.shape))

    # Tokenize function based on NLTK
    # Credit: http://nlpforhackers.io/recipe-text-clustering/
    def tokenize_nltk(self, text):
        self.tokenizer_counter += 1
        logger.debug("item:" + str(self.tokenizer_counter) +
                     "/" + str(self.tokenizer_len))
        toktok = ToktokTokenizer()
        # tokens =[toktok.tokenize(sent) for sent in sent_tokenize(text)]
        tokens = nltk.word_tokenize(text)
        # logger.debug("Number of tokens:" + str(len(tokens)))
        stems = [self.stemmer.stem(t) for t in tokens]
        return stems

    # A custom function to tokenize the text using spaCy
    # and convert to lemmas
    # credit: https://nicschrading.com/project/Intro-to-NLP-with-spaCy/
    def tokenize_spacy(self, text):
        self.tokenizer_counter += 1
        logger.debug("item:" + str(self.tokenizer_counter) +
                     "/" + str(self.tokenizer_len))
        # get the tokens using spaCy
        parser = English()
        tokens = parser(text)
        # logger.debug("Number of tokens:" + str(len(tokens)))
         # lemmatize
        # logger.debug("lemmatizing ...")
        lemmas = []
        for tok in tokens:
            lemmas.append(tok.lemma_.lower().strip()
                          if tok.lemma_ != "-PRON-" else tok.lower_)
        tokens = lemmas
        # stoplist the tokens
        # logger.debug("stoplist the tokens ...")
        tokens = [tok for tok in tokens if tok not in STOPLIST]

        # stoplist symbols
        # logger.debug("stoplist the symbols ...")
        tokens = [tok for tok in tokens if tok not in SYMBOLS]

        # remove large strings of whitespace
        # logger.debug("remove large strings of whitespace ...")
        # while "" in tokens:
        #     tokens.remove("")
        # while " " in tokens:
        #     tokens.remove(" ")
        # while "\n" in tokens:
        #     tokens.remove("\n")
        # while "\n\n" in tokens:
        #     tokens.remove("\n\n")
        return tokens

    # Word2Vec model
    # credit:
    # https://github.com/nadbordrozd/blog_stuff/blob/master/classification_w2v/benchmarking.ipynb
    def word2vec(self, data, max_features=MAX_FEATURES, min_df=5, n_jobs=W2V_N_THREADS, load=False):
        # train word2vec on all the texts - both training and test set
        # we're not using test labels, just texts so this is fine
        model_file = DATA_DIR + "/word2vec"
        if load:
            model = Word2Vec.load(model_file)
        else:
            logger.debug("training Word2Vec ...")
            start = time.time()
            model = Word2Vec(data, size=max_features,
                             min_count=min_df, workers=n_jobs)
            end = time.time() - start
            model.save(model_file)
            logger.debug("Word2vec training time:" + str(end))
        w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}
        return w2v

    # Use 1st time when to train word2vec model. After that load from disk
    def train_word2vec(self):
        data = self.combine_data[TEXT_COL]
        w2v = self.word2vec(data, load=False)
        return w2v

    # extract text features using word2vec and TF-IDF
    def text_features_w2v(self, data):
        w2v = self.word2vec(data, load=True)
        logger.debug("Extracting text features ...")
        tfidf = TfidfEmbeddingVectorizer(w2v)
        start = time.time()
        array = tfidf.fit_transform(data)
        end = time.time() - start
        logger.debug("tfidf size:" + str(array.shape))
        logger.debug("Text feature Transform time:" + str(end))
        return array

    # extract text features using TF-IDF and tokenizer
    def text_features(self, data):
        logger.debug("Extracting text features ...")
        if self.tf_idf_type == TF_IDF_NLTK:
            logger.debug("-- Using TF_IDF_NLTK tokenizer")
            tokenizer = self.tokenize_nltk
        elif self.tf_idf_type == TF_IDF_SPACY:
            logger.debug("-- Using TF_IDF_SPACY tokenizer")
            tokenizer = self.tokenize_spacy
        else:
            logger.debug("-- Using None tokenizer")
            tokenizer = None
        self.tokenizer_counter = 0
        self.tokenizer_len = len(data)
        tfidf = TfidfVectorizer(
            min_df=5, max_features=MAX_FEATURES, stop_words=STOPLIST,
            tokenizer=tokenizer, lowercase=True)
        # tfidf = TfidfVectorizer(
        #     min_df=5, max_features=None, stop_words=STOPLIST,
        #     tokenizer=tokenizer, lowercase=True)
        start = time.time()
        array = tfidf.fit_transform(data)
        end = time.time() - start
        logger.debug("tfidf size:" + str(array.shape))
        logger.debug("Text feature Transform time:" + str(end))
        return array

    # Use 1st time when to extract text features. After that load features
    # from disk
    def extract_text_features(self, load=False):
        data_tf = None
        if load:
            with open(self.tf_filename, 'rb') as infile:
                data_tf = pickle.load(infile)
        else:
            data = self.combine_data[TEXT_COL]
            if self.tf_idf_type == TF_IDF_W2C:
                logger.debug("Using TF_IDF and W2VEC")
                data_tf = self.text_features_w2v(data)
            else:
                logger.debug("Using TF_IDF")
                data_tf = self.text_features(data)
            if (self.use_lsa):
                # Feature reducion
                logger.debug("Feature reducion from:" +
                             str(data_tf.shape) + " to:" + str(LSA_FEATURES) + " ...")
                start = time.time()
                svd = TruncatedSVD(n_components=LSA_FEATURES, random_state=700)
                data_tf = svd.fit_transform(data_tf)
                data_tf = sparse.csr_matrix(data_tf)
                end = time.time() - start
                logger.debug("Feature reducion time:" + str(end))
            with open(self.tf_filename, 'wb') as outfile:
                pickle.dump(data_tf, outfile, pickle.HIGHEST_PROTOCOL)
        return data_tf

    def prepare_training_data_set(self):
        logger.debug("Preparing training data set ...")
        data_tf = self.extract_text_features(True)
        logger.debug("Text tf:" + str(type(data_tf)) +
                     " size:" + str(data_tf.shape))
        # data_tf = sparse.hstack([data_tf, genes_tf, variations_tf],
        # format="csr") => Variation lower score
        # data_tf = sparse.hstack([data_tf, genes_tf], format="csr")
        # print("combine matrix:", type(data_tf), " size:", data_tf.shape)
        return data_tf

    def build_model(self):
        # model = SVC(decision_function_shape='ovo',
        #             probability=True, random_state=250)
        # model = XGBClassifier(n_estimators=5, max_depth=5,
        #                       n_jobs=-1, silent=False)
        # specify your configurations as a dict

        model = LGBMClassifier(objective='multiclass',
                               n_estimators=N_ROUNDS,
                               learning_rate=0.01,
                               num_leaves=1024,
                               seed=12434,
                               nthread=-1, silent=False,
                               )
        return model

    def split_train_test(self):
        self.eval_id = self.eval_data['id']
        data_tf = self.prepare_training_data_set()
        train_len = len(self.train_data)
        self.train_set = data_tf[:train_len]
        self.eval_set = data_tf[train_len:]
        print("Train set:", self.train_set.shape,
              " eval set:", self.eval_set.shape)
        self.train_data["EAP"] = (self.train_data.author == "EAP") * 1
        self.train_data["HPL"] = (self.train_data.author == "HPL") * 1
        self.train_data["MWS"] = (self.train_data.author == "MWS") * 1
        self.target_vars = ["EAP", "HPL", "MWS"]
        # self.target = self.train_data[self.target_vars].values
        # self.target = self.train_data[LABEL]
        lbl = LabelEncoder()
        self.target = lbl.fit_transform(self.train_data[LABEL])
        X_train, X_test, Y_train, Y_test = train_test_split(
            self.train_set, self.target, train_size=0.7, random_state=324)
        logger.debug("X_train:" + str(X_train.shape) +
                     ". X_test:" + str(X_test.shape) + ". Y_train:" + str(Y_train.shape))
        return X_train, X_test, Y_train, Y_test

    def train_model(self):
        X_train, X_test, Y_train, Y_test = self.split_train_test()
        # create dataset for lightgbm
        # if you want to re-use data, remember to set free_raw_data=False
        lgb_train = lgb.Dataset(
            X_train, Y_train, max_bin=63, free_raw_data=False)
        lgb_eval = lgb.Dataset(
            X_test, Y_test, max_bin=63, reference=lgb_train, free_raw_data=False)

        # self.model = self.build_model()
        params = {
            # 'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'num_leaves': 1024,
            'learning_rate': LEARNING_RATE,
            'device': 'gpu',
            'gpu_use_dp': False,
            # 'sparse_threshold': 0.5,
            'verbose': 1
        }
        logger.debug("Training model ...")
        start = time.time()
        # self.model.fit(
        #     X_train, Y_train, eval_set=[(X_test, Y_test)],
        #     eval_metric="multi_logloss",
        #     early_stopping_rounds=EARLY_STOPPING,
        #     verbose=True,
        # )
        evals_result = {}
        self.model = lgb.train(params,
                               lgb_train,
                               valid_sets=lgb_eval,  # eval training data
                               valid_names=['val'],
                               num_boost_round=N_ROUNDS,
                               #    learning_rates=lambda iter: LEARNING_RATE * \
                               #    (0.99 ** iter),
                               early_stopping_rounds=EARLY_STOPPING,
                               evals_result=evals_result,
                               verbose_eval=EARLY_STOPPING,
                               )
        end = time.time() - start
        logger.debug("Done training model:" + str(end))
        # score = self.model.score(X_test, Y_test)
        # logger.debug(" score:" + str(score))

    # Predict evaluation data and save to CSV file
    def predict_save_eval(self):
        # predictions
        logger.debug("Eval data predicting ... ")
        data_eval = self.eval_set
        logger.debug("Eval size:" + str(data_eval.shape))
        start = time.time()
        y_pred = self.model.predict(
            data_eval, num_iteration=self.model.best_iteration)
        end = time.time() - start
        logger.debug("Export data")
        preds = pd.DataFrame(y_pred, columns=self.target_vars)
        eval_output = pd.concat([self.eval_id, preds], 1)
        today = str(dtime.date.today())
        logger.debug("Date:" + today)
        eval_output.to_csv(
            DATA_DIR + '/' + today + '-submission.csv.gz', index=False, float_format='%.5f',
            compression='gzip')


# ---------------- Utility classes -----------------------------------
# Tranform text using TF-IDF and word2vec
# credit:
# https://github.com/nadbordrozd/blog_stuff/blob/master/classification_w2v/benchmarking.ipynb


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.values())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        array = np.array([
            np.mean([self.word2vec[w] * self.word2weight[w]
                     for w in words if w in self.word2vec] or
                    [np.zeros(self.dim)], axis=0)
            for words in X
        ])
        return sparse.csr_matrix(array)

    def fit_transform(self, X):
        self.fit(X, None)
        return self.transform(X)

# Define a custom transformer to clean text using spaCy
# credit: https://nicschrading.com/project/Intro-to-NLP-with-spaCy/


class CleanTextTransformer(TransformerMixin):
    """
    Convert text to cleaned text
    """

    def transform(self, X, **transform_params):
        return [cleanText(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}

    # A custom function to clean the text before sending it into the
    # vectorizer
    def cleanText(self, text):
        # get rid of newlines
        text = text.strip().replace("\n", " ").replace("\r", " ")

        # replace twitter @mentions
        mentionFinder = re.compile(r"@[a-z0-9_]{1,15}", re.IGNORECASE)
        text = mentionFinder.sub("@MENTION", text)

        # replace HTML symbols
        text = text.replace("&amp;", "and").replace(
            "&gt;", ">").replace("&lt;", "<")

        # lowercase
        text = text.lower()
        return text


# -------------------- Main program ----------------
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
    option = 10
    classifier = SpookyAuthorIdentifer(LABEL, TF_IDF_NLTK, False)
    classifier.load_data()

    if option == 1:
        # Train word2vec model based on combine_data
        classifier.train_word2vec()
    elif option == 2:
        # Extract text features based on TF-IDF. This is the input to model
        # training
        classifier.extract_text_features()
    elif option == 10:
        # classifier.extract_text_features()
        classifier.train_model()
        classifier.predict_save_eval()
