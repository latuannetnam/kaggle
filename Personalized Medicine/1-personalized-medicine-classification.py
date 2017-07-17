# Personalized Medicine: Redefining Cancer Treatment
# - Predict the effect of Genetic Variants to enable Personalized Medicine
# - https://www.kaggle.com/c/msk-redefining-cancer-treatment/data
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
from xgboost.sklearn import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_decision_regions, plot_learning_curves, plot_confusion_matrix

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
# from spacy.en import English
# gensim
from gensim.models.word2vec import Word2Vec
from collections import Counter, defaultdict
# --------- System ---------
import datetime
import sys
from inspect import getsourcefile
import os.path
import re
import time
import string
import pickle
import warnings
# warnings.filterwarnings('ignore')
# %matplotlib inline
# Input data files are available in the DATA_DIR directory.
DATA_DIR = "data-temp"
LABEL = 'Class'
TEXT_COL = 'Text'
# A custom stoplist
STOPLIST = set(sw.words('english') +
               ["n't", "'s", "'m", "ca"] + list(ENGLISH_STOP_WORDS))
# List of symbols we don't care about
SYMBOLS = " ".join(string.punctuation).split(
    " ") + ["-----", "---", "...", "“", "”", "'ve"]

# Maximum number of features for text extraction
MAX_FEATURES = 2000
# Maximum number of thread to train word2vec model
W2V_N_THREADS = 4


class PersonalizedMedicineClassifier:
    def __init__(self, label):
        self.label = label
        # initialize stemmer for NLTK processing
        self.stemmer = PorterStemmer()
        # initilize spaCy parse for English language
        # self.parser = English()

    def load_data(self):
        # Load data.
        # Download from:
        # https://www.kaggle.com/c/msk-redefining-cancer-treatment/data
        train_text = pd.read_csv(DATA_DIR + "/training_text.csv",
                                 sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])
        train_class = pd.read_csv(DATA_DIR + "/training_variants.csv")
        test_text = pd.read_csv(DATA_DIR + "/test_text.csv", sep="\|\|",
                                engine="python", skiprows=1, names=["ID", "Text"])
        test_class = pd.read_csv(DATA_DIR + "/test_variants.csv")
        # combine train_text + train_class
        self.train_full = train_class.merge(
            train_text, how="inner", left_on="ID", right_on="ID")
        # train_full.head(5)
        # combine test_text + test_class
        self.test_full = test_class.merge(
            test_text, how="inner", left_on="ID", right_on="ID")
        # test_full.head(5)
        # Target
        o_features = self.test_full.columns.values
        # Store eval_id
        self.eval_data_id = self.test_full['ID'].values
        # Seperate input and label from train_data
        input_data = self.train_full[o_features]
        self.target = self.train_full[self.label]
        # target.describe()
        # Combine train + eval data
        self.combine_data = pd.concat(
            [input_data, self.test_full], keys=['train', 'eval'])

        # Tranform label
        # Label encode the targets
        labels = LabelEncoder()
        target_tf = labels.fit_transform(self.target)
        self.target = target_tf

    def transform_data(self):
        pass

    def check_null_data(self, data):
        # Get high percent of NaN data
        null_data = data.isnull()
        total = null_data.sum().sort_values(ascending=False)
        percent = (null_data.sum() / null_data.count()
                   ).sort_values(ascending=False)
        missing_data = pd.concat(
            [total, percent], axis=1, keys=['Total', 'Percent'])
        high_percent_miss_data = missing_data[missing_data['Percent'] > 0]
        # print(missing_data)
        print(high_percent_miss_data)
        miss_data_cols = high_percent_miss_data.index.values
        return miss_data_cols

    # Tokenize function based on NLTK
    # Credit: http://nlpforhackers.io/recipe-text-clustering/
    def tokenize_nltk(self, text):
        toktok = ToktokTokenizer()
        # tokens =[toktok.tokenize(sent) for sent in sent_tokenize(text)]
        tokens = nltk.word_tokenize(text)
        stems = [stemmer.stem(t) for t in tokens]
        # print("Number of tokens:", len(tokens))
        return stems

    # A custom function to tokenize the text using spaCy
    # and convert to lemmas
    # credit: https://nicschrading.com/project/Intro-to-NLP-with-spaCy/
    def tokenize_spacy(self, sample):

        # get the tokens using spaCy
        tokens = self.parser(sample)

        # lemmatize
        lemmas = []
        for tok in tokens:
            lemmas.append(tok.lemma_.lower().strip()
                          if tok.lemma_ != "-PRON-" else tok.lower_)
        tokens = lemmas

        # stoplist the tokens
        tokens = [tok for tok in tokens if tok not in STOPLIST]

        # stoplist symbols
        tokens = [tok for tok in tokens if tok not in SYMBOLS]

        # remove large strings of whitespace
        while "" in tokens:
            tokens.remove("")
        while " " in tokens:
            tokens.remove(" ")
        while "\n" in tokens:
            tokens.remove("\n")
        while "\n\n" in tokens:
            tokens.remove("\n\n")
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
            print("training Word2Vec ...")
            start = time.time()
            model = Word2Vec(data, size=max_features,
                             min_count=min_df, workers=n_jobs)
            end = time.time() - start
            model.save(model_file)
            print("Word2vec training time:", end)
        w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}
        return w2v

    # Use 1st time when to train word2vec model. After that load from disk
    def train_word2vec(self):
        data = self.combine_data[TEXT_COL]
        w2v = word2vec(data, load=False)
        return w2v

    # extract text features using word2vec and TF-IDF
    def text_features_w2v(self, data, load=False):
        filename = DATA_DIR + "/tfidf.npz"
        if load:
            with open(filename, 'rb') as infile:
                array = pickle.load(infile)
        else:
            w2v = self.word2vec(data, load=True)
            print("Extracting text features ...")
            tfidf = TfidfEmbeddingVectorizer(w2v)
            start = time.time()
            array = tfidf.fit_transform(data)
            end = time.time() - start
            with open(filename, 'wb') as outfile:
                pickle.dump(array, outfile, pickle.HIGHEST_PROTOCOL)
            print("Text feature Transform time:", end)
        return array

    # Use 1st time when to extract text features. After that load features
    # from disk
    def extract_text_features(self):
        data = self.combine_data[TEXT_COL]
        self.text_features_w2v(data, load=False)

    def build_model(self):
        # model = SVC(decision_function_shape='ovo',
        #             probability=True, random_state=250)
        model = XGBClassifier(n_estimators=500, max_depth=5, n_jobs = -1)
        return model

    def split_train_test(self):
        data = self.combine_data[TEXT_COL]
        data_tf = self.text_features_w2v(data, load=True)
        train_len = len(self.train_full)
        self.train_set = data_tf[:train_len]
        self.eval_set = data_tf[train_len:]
        print("Train set:", self.train_set.shape,
              " eval set:", self.eval_set.shape)
        X_train, X_test, Y_train, Y_test = train_test_split(
            self.train_set, self.target, train_size=0.7, random_state=324)
        print("train size:", X_train.shape)
        print("test size:", X_test.shape)
        return X_train, X_test, Y_train, Y_test

    def train_model(self):
        self.model = self.build_model()
        X_train, X_test, Y_train, Y_test = self.split_train_test()
        print("Training model ...")
        start = time.time()
        self.model.fit(X_train, Y_train)
        end = time.time() - start
        print("Done training model:", end)
        score = self.model.score(X_test, Y_test)
        print(" score:", score)

    # Predict evaluation data and save to CSV file
    def predict_save_eval(self):
        # predictions
        print("Eval data predicting ... ")
        data_eval = self.eval_set
        start = time.time()
        y_pred = self.model.predict_proba(data_eval)
        end = time.time() - start
        print("Export data")
        # tweaking the submission file as required
        # credit: https://www.kaggle.com/punyaswaroop12/gbm-starter-top-40
        subm_file = pd.DataFrame(y_pred)
        subm_file['id'] = self.eval_data_id
        subm_file.columns = ['class1', 'class2', 'class3', 'class4',
                             'class5', 'class6', 'class7', 'class8', 'class9', 'id']
        subm_file.to_csv(DATA_DIR + "/submission_v3.csv", index=False)
        subm_file.head(5)


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
classifier = PersonalizedMedicineClassifier(LABEL)
classifier.load_data()
classifier.train_model()
classifier.predict_save_eval()
