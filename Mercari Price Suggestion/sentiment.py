'''
Based on https://www.kaggle.com/knowledgegrappler/a-simple-nn-solution-with-keras-0-48611-pl

Optimization:
- Word lemmatization
- Sentiment analysis for descriptions
- Add description length
- Guess from description whether item had pictures
- Minor fixes

Changelog 2017-11-29, 17:34:
- Lemmatize with correct POS (takes too long :-/)
- More GRU units for description
- Try Nadam optimizer

Changelog 2017-11-30, 23:57:
- Remove lemmatization for performance reasons
- Guess item quantity from name
'''

import math
import re
import sys
import time
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
from nltk import pos_tag
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from keras.utils import plot_model
LOAD_MODEL = False
n_threads = 4  # psutil.cpu_count()
print('Using {} threads when doing multiprocessing'.format(n_threads))


def flatten(l): return [item for sublist in l for item in sublist]


def rmsle(y, y_pred):
    assert len(y) == len(y_pred)
    to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1))
              ** 2.0 for i, pred in enumerate(y_pred)]
    return (sum(to_sum) * (1.0 / len(y))) ** 0.5


def handle_missing(dataset):
    dataset.category_name.fillna(value="missing", inplace=True)
    dataset.brand_name.fillna(value="missing", inplace=True)
    dataset.item_description.fillna(value="missing", inplace=True)
    return (dataset)


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN  # pos_tag default


def _get_lemma_desc(args):
    data, index = args
    lmtzr = WordNetLemmatizer()
    lemmas = []
    for s in data:
        words = [w.lower() for w in word_tokenize(s)]
        tags = pos_tag(words)
        lemmatized = []
        for i, w in enumerate(words):
            if not w.isalpha():
                continue
            lemmatized.append(lmtzr.lemmatize(
                w, get_wordnet_pos(tags[i][1])).lower())
        lemmas.append(' '.join(lemmatized))
    return pd.Series(lemmas, index=index)


def get_lemma_desc(data, index):
    p = Pool(processes=n_threads)
    n = math.ceil(len(data) / n_threads)
    lemmas = p.map(_get_lemma_desc, [
                   (data[i:i + n], index[i:i + n]) for i in range(0, len(data), n)])
    return np.array(flatten(lemmas))


def may_have_pictures(descriptions):
    matches = []
    for desc in descriptions:
        match = 0
        for r in pic_word_re:
            if r.search(desc) is not None:
                match = 1
                continue
        matches.append(match)
    return np.array(matches)


def get_keras_data(dataset, maxlengths):
    from keras.preprocessing.sequence import pad_sequences
    X = {
        'name': pad_sequences(dataset.seq_name, maxlen=maxlengths['MAX_NAME_SEQ']),
        'item_desc': pad_sequences(dataset.seq_item_description, maxlen=maxlengths['MAX_ITEM_DESC_SEQ']),
        'brand_name': np.array(dataset.brand_name), 'category_name': np.array(dataset.category_name),
        'item_condition': np.array(dataset.item_condition_id),
        'num_vars': np.array(dataset[["shipping", "desc_sentiment", "desc_len", "may_have_pictures"]])
    }
    return X


def rmsle_cust(y_true, y_pred):
    from keras import backend as K
    first_log = K.log(K.clip(y_pred, K.epsilon(), None) + 1.)
    second_log = K.log(K.clip(y_true, K.epsilon(), None) + 1.)
    return K.sqrt(K.mean(K.square(first_log - second_log), axis=-1))


def get_model(X_train, max_vocabulary):
    from keras.layers import Input, Dropout, Dense, BatchNormalization, Activation, concatenate, GRU, Embedding, Flatten, BatchNormalization
    from keras.models import Model

    # params
    dr_r = 0.1

    # Inputs
    name = Input(shape=[X_train["name"].shape[1]], name="name")
    item_desc = Input(shape=[X_train["item_desc"].shape[1]], name="item_desc")
    brand_name = Input(shape=[1], name="brand_name")
    category_name = Input(shape=[1], name="category_name")
    item_condition = Input(shape=[1], name="item_condition")
    num_vars = Input(shape=[X_train["num_vars"].shape[1]], name="num_vars")

    # Embeddings layers
    emb_name = Embedding(max_vocabulary['MAX_TEXT'], 50)(name)
    emb_item_desc = Embedding(max_vocabulary['MAX_TEXT'], 50)(item_desc)
    emb_brand_name = Embedding(max_vocabulary['MAX_BRAND'], 10)(brand_name)
    emb_category_name = Embedding(
        max_vocabulary['MAX_CATEGORY'], 10)(category_name)
    emb_item_condition = Embedding(
        max_vocabulary['MAX_CONDITION'], 5)(item_condition)

    # rnn layer
    rnn_layer1 = GRU(24)(emb_item_desc)
    rnn_layer2 = GRU(8)(emb_name)

    # main layer
    main_l = concatenate([
        Flatten()(emb_brand_name),
        Flatten()(emb_category_name),
        Flatten()(emb_item_condition),
        rnn_layer1,
        rnn_layer2,
        num_vars
    ])
    main_l = Dropout(dr_r)(Dense(256)(main_l))
    main_l = Dropout(dr_r)(Dense(128)(main_l))
    main_l = Dropout(dr_r)(Dense(64)(main_l))

    # output
    output = Dense(1, activation="linear")(main_l)

    # model
    model = Model([name, item_desc, brand_name, category_name,
                   item_condition, num_vars], output)
    model.compile(loss="mse", optimizer="nadam", metrics=["mae", rmsle_cust])
    print(model.summary())
    # Plot the model
    plot_model(model, show_shapes=True,
               to_file='data-temp/model-sentiment.png')
    return model


def _get_sentiment(data):
    sia = SentimentIntensityAnalyzer()
    scores = []
    for s in data:
        scores.append(sia.polarity_scores(s)['compound'])
    return np.array(scores)


def get_sentiment(data):
    p = Pool(processes=n_threads)
    n = math.ceil(len(data) / n_threads)
    scores = p.map(_get_sentiment, [data[i:i + n]
                                    for i in range(0, len(data), n)])
    return np.array(flatten(scores))


def get_qntys(data):
    qnty_matches = []
    qnty_re = [r'(\d+) ?x [^\d]', r'(\d+) ?pairs?']
    for r in qnty_re:
        qnty_matches.append(data.name.str.extract(
            r, flags=re.IGNORECASE, expand=False).dropna())
    return pd.concat(qnty_matches).reset_index().drop_duplicates(subset='index', keep='last').set_index('index')


if __name__ == '__main__':
    start_time = time.time()

    engine = 'python' if sys.platform == 'win32' else 'c'

    # LOAD DATA
    print("Loading data...")
    train = pd.read_table("data-temp/train.tsv", engine=engine).sample(frac=1.0)
    test = pd.read_table("data-temp/test.tsv", engine=engine).sample(frac=1.0)

    print('[{}] Finished loading data'.format(time.time() - start_time))

    # HANDLE MISSING VALUES
    print("Handling missing values...")
    train = handle_missing(train)
    test = handle_missing(test)

    print('[{}] Finished handling missing values'.format(
        time.time() - start_time))

    # Lemmatize
    # Lemmatization with proper POS tags takes too long for the kernel --> skip lemmatization for now
    #train['desc_lemmas'] = get_lemma_desc(train.item_description, train.index)
    #test['desc_lemmas'] = get_lemma_desc(test.item_description, test.index)
    train['desc_lemmas'] = train.item_description
    test['desc_lemmas'] = test.item_description

    print('[{}] Finished lemmatization (train: {}, test: {})'.format(
        time.time() - start_time, len(train.desc_lemmas), len(test.desc_lemmas)))

    # Estimate quantity
    train['qnty'] = get_qntys(train)
    test['qnty'] = get_qntys(test)
    train.fillna(value={'qnty': 1}, inplace=True)
    test.fillna(value={'qnty': 1}, inplace=True)
    train['qnty'] = MinMaxScaler().fit_transform(train['qnty'].reshape(-1, 1))
    test['qnty'] = MinMaxScaler().fit_transform(test['qnty'].reshape(-1, 1))
    print('[{}] Finished estimating quantities (train: {}, test: {})'.format(time.time(
    ) - start_time, train[train.qnty > train.qnty.min()].size, test[test.qnty > test.qnty.min()].size))

    # PROCESS CATEGORICAL DATA
    print("Handling categorical variables...")
    le = LabelEncoder()

    le.fit(np.hstack([train.category_name, test.category_name]))
    train.category_name = le.transform(train.category_name)
    test.category_name = le.transform(test.category_name)

    le.fit(np.hstack([train.brand_name, test.brand_name]))
    train.brand_name = le.transform(train.brand_name)
    test.brand_name = le.transform(test.brand_name)
    del le

    # PROCESS TEXT: RAW
    from keras.preprocessing.text import Tokenizer

    print("Text to seq process...")
    raw_text = np.hstack([train.desc_lemmas.str.lower(), test.desc_lemmas.str.lower(
    ), train.name.str.lower(), test.name.str.lower()])

    print("Fitting tokenizer...")
    tok_raw = Tokenizer()
    tok_raw.fit_on_texts(raw_text)
    print("Transforming text to seq...")

    train["seq_item_description"] = tok_raw.texts_to_sequences(
        train.desc_lemmas.str.lower())
    test["seq_item_description"] = tok_raw.texts_to_sequences(
        test.desc_lemmas.str.lower())
    train["seq_name"] = tok_raw.texts_to_sequences(train.name.str.lower())
    test["seq_name"] = tok_raw.texts_to_sequences(test.name.str.lower())

    # SEQUENCES VARIABLES ANALYSIS
    max_name_seq = np.max([np.max(train.seq_name.apply(lambda x: len(x))), np.max(
        test.seq_name.apply(lambda x: len(x)))])
    max_seq_item_description = np.max([np.max(train.seq_item_description.apply(
        lambda x: len(x))), np.max(test.seq_item_description.apply(lambda x: len(x)))])

    # EMBEDDINGS MAX VALUE
    max_vocabulary = {
        'MAX_NAME_SEQ': 10,
        'MAX_ITEM_DESC_SEQ': 100,
        'MAX_TEXT': np.unique(flatten(np.concatenate([train.seq_item_description, test.seq_item_description, test.seq_name, train.seq_name]))).shape[0] + 1,
        'MAX_CATEGORY': np.unique(np.concatenate([train.category_name, test.category_name])).shape[0] + 1,
        'MAX_BRAND': np.unique(np.concatenate([train.brand_name, test.brand_name])).shape[0] + 1,
        'MAX_CONDITION': np.unique(np.concatenate([train.item_condition_id, test.item_condition_id])).shape[0] + 1
    }

    print(max_vocabulary)

    # SCALE target variable
    train["target"] = np.log(train.price + 1)
    target_scaler = MinMaxScaler(feature_range=(-1, 1))
    train["target"] = target_scaler.fit_transform(train.target.reshape(-1, 1))

    print('[{}] Finished additional preprocessing'.format(
        time.time() - start_time))

    # Add sentiment scores
    print('Getting item description\' sentiment for train set.')
    train['desc_sentiment'] = get_sentiment(train.item_description)

    print('Getting item description\' sentiment for test set.')
    test['desc_sentiment'] = get_sentiment(test.item_description)

    print('[{}] Finished sentiment analysis'.format(time.time() - start_time))

    # Add description length
    idx_split = len(train.seq_item_description)
    train['desc_len'] = np.array(
        list(map(lambda d: len(d), train.seq_item_description)))
    test['desc_len'] = np.array(
        list(map(lambda d: len(d), test.seq_item_description)))
    scaler = MinMaxScaler()
    all_scaled = scaler.fit_transform(np.concatenate(
        [train['desc_len'].reshape(-1, 1), test['desc_len'].reshape(-1, 1)]))
    train['desc_len'], test['desc_len'] = all_scaled[:idx_split], all_scaled[idx_split:]

    # Add whether it may have pictures
    pic_word_re = [re.compile(r, re.IGNORECASE) for r in [
        r'(see(n)?)?( in| the| my) (picture(s)?|photo(s)?)']]

    train['may_have_pictures'] = pd.Series(may_have_pictures(
        train.item_description), index=train.index).astype('category')
    test['may_have_pictures'] = pd.Series(may_have_pictures(
        test.item_description), index=test.index).astype('category')

    print('Found {} of {} items that potentially have pictures'.format(len(
        [x for x in train['may_have_pictures'] if x == 1]), len(train['item_description'])))

    # EXTRACT DEVELOPTMENT TEST
    dtrain, dvalid = train_test_split(train, random_state=123, train_size=0.95)
    print(dtrain.shape)
    print(dvalid.shape)

    # KERAS DATA DEFINITION
    X_train = get_keras_data(dtrain, max_vocabulary)
    X_valid = get_keras_data(dvalid, max_vocabulary)
    X_test = get_keras_data(test, max_vocabulary)

    print('[{}] Finished additional preprocessing and model initialization'.format(
        time.time() - start_time))
    BATCH_SIZE = 5000

    if LOAD_MODEL:
        from keras.models import load_model
        model = load_model('model1.h5', custom_objects={
                           'rmsle_cust': rmsle_cust})
        print('Loaded model.')
    else:
        # FITTING THE MODEL
        epochs = 2

        model = get_model(X_train, max_vocabulary)
        model.fit(X_train, dtrain.target, epochs=epochs, batch_size=BATCH_SIZE,
                  validation_data=(X_valid, dvalid.target), verbose=1)

        model.save('model2.h5')

        print('[{}] Finished training'.format(time.time() - start_time))

    # EVALUATE THE MODEL ON DEV TEST
    #val_preds = model.predict(X_valid)
    #val_preds = target_scaler.inverse_transform(val_preds)
    #val_preds = np.exp(val_preds) - 1

    #mean_absolute_error, mean_squared_log_error
    #y_true = np.array(dvalid.price.values)
    #y_pred = val_preds[:, 0]
    #v_rmsle = rmsle(y_true, y_pred)
    #print("RMSLE error on dev test: " + str(v_rmsle))

    #print('[{}] Finished evaluation'.format(time.time() - start_time))

    # CREATE PREDICTIONS
    # preds = model.predict(X_train, batch_size=BATCH_SIZE)
    # preds = target_scaler.inverse_transform(preds)
    # preds = np.exp(preds) - 1
    # train_preds = dtrain[["train_id"]]
    # train_preds["price"] = preds
    # train_preds.to_csv("./train_preds2.csv", index=False)

    preds = model.predict(X_test, batch_size=BATCH_SIZE)
    preds = target_scaler.inverse_transform(preds)
    preds = np.exp(preds) - 1
    submission = test[["test_id"]]
    submission["price"] = preds
    submission.to_csv("data-temp/submission2.csv", index=False)

    print('[{}] Finished predictions and saving'.format(
        time.time() - start_time))
