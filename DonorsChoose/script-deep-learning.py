'''
Based on https://www.kaggle.com/CVxTz/keras-baseline-feature-hashing-cnn
I add some new functions.
1. Do some data preprocessing.
2. Use crawl-300d-2M.vec
3. Use GRU before CNN layer
'''
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "data-temp/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
# print(os.listdir("data-temp"))
os.environ['PYTHONHASHSEED'] = '10000'
np.random.seed(10001)
import random
import tensorflow as tf
random.seed(10002)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=6, inter_op_parallelism_threads=5)
from keras import backend

tf.set_random_seed(10003)
backend.set_session(tf.Session(graph=tf.get_default_graph(), config=session_conf))
# Any results you write to the current directory are saved as output.
print("Load training data ...")
train = pd.read_csv("data-temp/train.csv", low_memory=False)
test = pd.read_csv("data-temp/test.csv", low_memory=False)
resources = pd.read_csv("data-temp/resources.csv", low_memory=False)
train = train.sort_values(by="project_submitted_datetime")
print("Processing data ...")
teachers_train = list(set(train.teacher_id.values))
teachers_test = list(set(test.teacher_id.values))
inter = set(teachers_train).intersection(teachers_test)

char_cols = ['project_subject_categories', 'project_subject_subcategories',
       'project_title', 'project_essay_1', 'project_essay_2',
       'project_essay_3', 'project_essay_4', 'project_resource_summary']
       

#https://www.kaggle.com/mmi333/beat-the-benchmark-with-one-feature
resources['total_price'] = resources.quantity * resources.price

mean_total_price = pd.DataFrame(resources.groupby('id').total_price.mean()) 
sum_total_price = pd.DataFrame(resources.groupby('id').total_price.sum()) 
count_total_price = pd.DataFrame(resources.groupby('id').total_price.count())
mean_total_price['id'] = mean_total_price.index
sum_total_price['id'] = mean_total_price.index
count_total_price['id'] = mean_total_price.index

def create_features(df):
    

    df = pd.merge(df, mean_total_price, on='id')
    df = pd.merge(df, sum_total_price, on='id')
    df = pd.merge(df, count_total_price, on='id')
    df['year'] = df.project_submitted_datetime.apply(lambda x: x.split("-")[0])
    df['month'] = df.project_submitted_datetime.apply(lambda x: x.split("-")[1])
    for col in char_cols:
        df[col] = df[col].fillna("NA")
    df['text'] = df.apply(lambda x: " ".join(x[col] for col in char_cols), axis=1)
    return df
print("Creating features ...")
train = create_features(train)
test = create_features(test)

cat_features = ["teacher_prefix", "school_state", "year", "month", "project_grade_category", "project_subject_categories", "project_subject_subcategories"]
#"teacher_id", 
num_features = ["teacher_number_of_previously_posted_projects", "total_price_x", "total_price_y", "total_price"]
cat_features_hash = [col+"_hash" for col in cat_features]

max_size=15000#0
def feature_hash(df, max_size=max_size):
    for col in cat_features:
        df[col+"_hash"] = df[col].apply(lambda x: hash(x)%max_size)
    return df

print("Feature hashing ...")
train = feature_hash(train)
test = feature_hash(test)
print("Feature scaling ...")
from sklearn.preprocessing import StandardScaler
#from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing import text, sequence
import re

max_features = 100000#50000
maxlen = 300
scaler = StandardScaler()
X_train_num = scaler.fit_transform(train[num_features])
X_test_num = scaler.transform(test[num_features])
X_train_cat = np.array(train[cat_features_hash], dtype=np.int)
X_test_cat = np.array(test[cat_features_hash], dtype=np.int)

print("Text tokenizing ...")
tokenizer = text.Tokenizer(num_words=max_features)

def preprocess1(string):
    '''
    :param string:
    :return:
    '''
    #去掉一些特殊符号
    string = re.sub(r'(\")', ' ', string)
    string = re.sub(r'(\r)', ' ', string)
    string = re.sub(r'(\n)', ' ', string)
    string = re.sub(r'(\r\n)', ' ', string)
    string = re.sub(r'(\\)', ' ', string)
    string = re.sub(r'\t', ' ', string)
    string = re.sub(r'\:', ' ', string)
    string = re.sub(r'\"\"\"\"', ' ', string)
    string = re.sub(r'_', ' ', string)
    string = re.sub(r'\+', ' ', string)
    string = re.sub(r'\=', ' ', string)

    return string

train["text"]=train["text"].apply(preprocess1)
test["text"]=test["text"].apply(preprocess1)

tokenizer.fit_on_texts(train["text"].tolist()+test["text"].tolist())
list_tokenized_train = tokenizer.texts_to_sequences(train["text"].tolist())
list_tokenized_test = tokenizer.texts_to_sequences(test["text"].tolist())
X_train_words = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
X_test_words = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)


X_train_target = train.project_is_approved
print("Processing pre-train embedding vector ...")
#data-temp/fatsttext-common-crawl/crawl-300d-2M/*
EMBEDDING_FILE = '/home/latuan/Programming/machine-learning/data/crawl-300d-2M.vec'
embed_size=300
embeddings_index = {}
with open(EMBEDDING_FILE,encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print("Creating embedding matrix ...")
word_index = tokenizer.word_index
#prepare embedding matrix
num_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


from keras.layers import Input, Dense, Embedding, Flatten, concatenate, Dropout, Convolution1D, \
GlobalMaxPool1D,SpatialDropout1D,CuDNNGRU,Bidirectional,PReLU,GRU
from keras.models import Model
from keras import optimizers
from keras import backend as K
from keras.utils import plot_model
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


def get_model3():
    input_cat = Input((len(cat_features_hash), ))
    input_num = Input((len(num_features), ))
    input_words = Input((maxlen, ))
    
    x_cat = Embedding(max_size, 10)(input_cat)
    
    x_cat = SpatialDropout1D(0.3)(x_cat)
    x_cat = Flatten()(x_cat)
    
    x_words = Embedding(max_features, 300,
                            weights=[embedding_matrix],
                            trainable=False)(input_words)
    x_words = SpatialDropout1D(0.3)(x_words)
    x_words =Bidirectional(GRU(50, return_sequences=True))(x_words)
    x_words = Convolution1D(100, 3, activation="relu")(x_words)
    x_words = GlobalMaxPool1D()(x_words)

    
    x_cat = Dense(100, activation="relu")(x_cat)
    x_num = Dense(100, activation="relu")(input_num)

    x = concatenate([x_cat, x_num, x_words])

    x = Dense(50, activation="relu")(x)
    x = Dropout(0.25)(x)
    predictions = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=[input_cat, input_num, input_words], outputs=predictions)
    model.compile(optimizer=optimizers.Adam(0.0005, decay=1e-6),
              loss='binary_crossentropy',
              metrics=['accuracy', auc])

    return model
print("Building model ...")
model = get_model3()
print(model.summary())
plot_model(model, show_shapes=True, to_file='data-temp/model-v2.png')
# model = get_model4()
# model = get_model3_v2()
from keras.callbacks import *
from sklearn.metrics import roc_auc_score

file_path='data-temp/simpleRNN3.h5'
checkpoint = ModelCheckpoint(file_path, monitor='val_auc', verbose=True, save_best_only=True, save_weights_only=True,
                                     mode='max')

early = EarlyStopping(monitor="val_auc", mode="max", patience=2)
lr_reduced = ReduceLROnPlateau(monitor='val_auc',
                               factor=0.1,
                               patience=2,
                               verbose=1,
                               epsilon=1e-4,
                               mode='max')
callbacks_list = [checkpoint, early, lr_reduced]
print("Training ...")
history = model.fit([X_train_cat, X_train_num, X_train_words], X_train_target, validation_split=0.1,
                    verbose=True,callbacks=callbacks_list,
          epochs=20, batch_size=256)
del X_train_cat, X_train_num, X_train_words,X_train_target
model.load_weights(file_path)
print("Predicting ...")
pred_test = model.predict([X_test_cat, X_test_num, X_test_words], batch_size=2000)

test["project_is_approved"] = pred_test
test[['id', 'project_is_approved']].to_csv("data-temp/gru_cnn_submission.csv", index=False)
print("Done")