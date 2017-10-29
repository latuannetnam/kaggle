# Porto Seguro’s Safe Driver Prediction
# Credit to:
# https://www.kaggle.com/arthurtok/interactive-porto-insights-a-plot-ly-tutorial
# https://www.kaggle.com/anokas/simple-xgboost-btb-0-27
# https://www.kaggle.com/rshally/porto-xgb-lgb-kfold-lb-0-282
# https://www.kaggle.com/akashdeepjassal/simple-keras-mlp/code
# https://www.kaggle.com/pnagel/keras-starter/code
# https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras
# https://www.kaggle.com/kostya17/simple-approach-to-handle-missing-values
# https://www.kaggle.com/camnugent/deep-neural-network-insurance-claims-0-268
# https://www.kaggle.com/rspadim/gini-keras-callback-earlystopping-validation
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

# LightGBM
from lightgbm import LGBMClassifier, LGBMRegressor

# Keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import model_from_json

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


def gini(y, pred):
    g = np.asarray(np.c_[y, pred, np.arange(len(y))], dtype=np.float)
    g = g[np.lexsort((g[:, 2], -1 * g[:, 1]))]
    gs = g[:, 0].cumsum().sum() / g[:, 0].sum()
    gs -= (len(y) + 1) / 2.
    return gs / len(y)


def gini_normalized(y, pred):
    return gini(y, pred) / gini(y, y)


def gini_xgb(y, pred):
    return 'gini', gini_normalized(Y, pred)


def gini_lgb(y, pred):
    score = gini_normalized(Y, pred)
    # score = gini(y, pred)
    return 'gini', score, True

# FROM https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/41108


def keras_auc(y_true, y_pred):
    score, up_opt = tf.metrics.auc(y_true, y_pred)
    # score, up_opt = tf.contrib.metrics.streaming_auc(y_pred, y_true)
    keras.backend.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
    return score


def keras_gini(y_true, y_pred):   # Not work yet
    score, up_opt = tf.metrics.auc(y_true, y_pred)
    # score, up_opt = tf.contrib.metrics.streaming_auc(y_pred, y_true)
    keras.backend.get_session().run(tf.local_variables_initializer())
    with tf.control_dependencies([up_opt]):
        score = tf.identity(score)
        score = score * 2 - 1
    return score


class keras_gini(keras.callbacks.Callback):
    def __init__(self, validation_data, patience=0, save_best=True, verbose=True):
        self.validation_data = validation_data
        self.patience = patience
        self.save_best = save_best
        self.verbose = verbose
        self.history = {}
        self.history['gini'] = []

    def eval_metric(self):
        # print("validation len:", len(self.validation_data))
        x_val = self.validation_data[0]
        y_true = self.validation_data[1]
        y_pred = self.model.predict(x_val, batch_size=KERAS_PREDICT_BATCH_SIZE)
        score = gini_normalized(y_true[:, 1], y_pred[:, 1])
        return score

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.best = -np.Inf
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs={}):
        score = self.eval_metric()

        self.history['gini'].append(score)
        if np.greater(score, self.best):
            self.best = score
            self.best_epoch = epoch + 1
            self.wait = 0
            if self.save_best:
                self.model.save_weights(model_path, overwrite=True)
        else:
            if self.patience > 0:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch + 1
                    self.model.stop_training = True
        if self.verbose:
            logger.debug("Eval for epoch " + str(epoch + 1) + ":" + str(score) +
                         ". Best score:" + str(self.best) + ". Best epoch:" + str(self.best_epoch))

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            logger.debug("Epoch " + str(self.stopped_epoch) + " early stopping" +
                         ". Best score:" + str(self.best) + ". Best epoch:" + str(self.best_epoch))


# ---------------- Main -------------------------
if __name__ == "__main__":
    logger.info("Running ..")
    # Load data. Download
    # from:https://www.kaggle.com/c/nyc-taxi-trip-duration/data
    logger.info("Loading data ...")
    train_data = pd.read_csv(DATA_DIR + "/train.csv", na_values="-1")
    eval_data = pd.read_csv(DATA_DIR + "/test.csv")

    logger.debug("train size:" + str(train_data.shape) +
                 " test size:" + str(eval_data.shape))
    label = 'target'
    target = train_data[label]
    eval_id = eval_data['id']

    logger.info("Filling NaN values ...")
    # Drop high NaN columns
    # Credit to:
    # https://www.kaggle.com/kostya17/simple-approach-to-handle-missing-values
    # train_data.drop(["id", "target", "ps_car_03_cat",
    #                  "ps_car_05_cat"], axis=1, inplace=True)
    # eval_data.drop(["id", "ps_car_03_cat", "ps_car_05_cat"],
    #                axis=1, inplace=True)
    train_data.drop(["id", "target"], axis=1, inplace=True)
    eval_data.drop(["id"], axis=1, inplace=True)
    # Fill NaN value
    # "cat" - categorical: fill missing values with mode value of particular column
    # "bin" - binary: fill missing values with mode value of particular column
    # all other - (continuous or ordinal): fill with mean value of particular
    # column
    cat_cols = [col for col in train_data.columns if 'cat' in col]
    bin_cols = [col for col in train_data.columns if 'bin' in col]
    con_cols = [
        col for col in train_data.columns if col not in bin_cols + cat_cols]

    for col in cat_cols:
        train_data[col].fillna(value=train_data[col].mode()[0], inplace=True)
        eval_data[col].fillna(value=eval_data[col].mode()[0], inplace=True)

    for col in bin_cols:
        train_data[col].fillna(value=train_data[col].mode()[0], inplace=True)
        eval_data[col].fillna(value=eval_data[col].mode()[0], inplace=True)

    for col in con_cols:
        train_data[col].fillna(value=train_data[col].mean(), inplace=True)
        eval_data[col].fillna(value=eval_data[col].mean(), inplace=True)

    # transform categories feature to one-hot-encode
    # credit: https://www.kaggle.com/camnugent/deep-neural-network-insurance-claims-0-268
    # one hot encode the categoricals
    logger.info("One hot encoding ...")
    merged_dat = pd.concat([train_data, eval_data], axis=0)
    cat_features = [col for col in merged_dat.columns if col.endswith('cat')]
    for column in cat_features:
        temp = pd.get_dummies(pd.Series(merged_dat[column]))
        merged_dat = pd.concat([merged_dat, temp], axis=1)
        merged_dat = merged_dat.drop([column], axis=1)

    train_data = merged_dat[:train_data.shape[0]].astype(np.float32)
    eval_data = merged_dat[train_data.shape[0]:].astype(np.float32)

    # prepare training data
    logger.info("Preparing training data ...")

    # Compute class weigth to balance label
    cw = class_weight.compute_class_weight(
        'balanced', np.unique(target), target)
    class_weight_dict = dict(enumerate(cw))

    # Split train/test set
    # X_train, X_test, Y_train, Y_test = train_test_split(
    #     data, target, train_size=0.85, random_state=1234, stratify=target)
    X_train = train_data
    Y_train = target
    logger.debug("X_train:" + str(X_train.shape) +
                 " Y_train:" + str(Y_train.shape))
    logger.debug("eval_data:" + str(eval_data.shape))

    #   " X_test:", X_test.shape, " Y_test:", Y_test.shape)

    # Scaling features
    scaler = StandardScaler()
    scaler.fit(X_train)
    # Transform train data
    X_train = scaler.transform(X_train)
    # X_test = scaler.transform(X_test)

    # Transform label to categorial
    Y_train = np_utils.to_categorical(Y_train.values)
    # Y_test = np_utils.to_categorical(Y_test.values)
    # print("Y-train:", Y_train[:5])

    # Model definition
    logger.info("Model definition")
    KERAS_LEARNING_RATE = 0.001
    KERAS_N_ROUNDS = 10000
    KERAS_BATCH_SIZE = 64
    KERAS_NODES = 1024
    KERAS_LAYERS = 6
    KERAS_DROPOUT_RATE = 0.5
    # KERAS_REGULARIZER = KERAS_LEARNING_RATE/10
    KERAS_REGULARIZER = 0
    KERAS_VALIDATION_SPLIT = 0.2
    KERAS_EARLY_STOPPING = 10
    KERAS_MAXNORM = 3
    KERAS_PREDICT_BATCH_SIZE = 1024
    VERBOSE = True
    model_weight_path = DATA_DIR + "/model_weight.h5"
    model_path = DATA_DIR + "/model.json"
    random_state = 12343
    n_features = X_train.shape[1]
    decay = KERAS_LEARNING_RATE / KERAS_N_ROUNDS
    nodes = KERAS_NODES
    dropout = KERAS_DROPOUT_RATE
    # , kernel_regularizer=keras.regularizers.l2(KERAS_REGULARIZER)
    # create model
    model = Sequential()
    model.add(Dropout(KERAS_DROPOUT_RATE,  input_shape=(
        n_features, ), seed=random_state))
    # model.add(Dense(KERAS_NODES, input_shape=(n_features, ),
    #                 activation='relu', kernel_constraint=keras.constraints.maxnorm(KERAS_MAXNORM)))
    # model.add(Dropout(KERAS_DROPOUT_RATE, seed=random_state))
    # nodes = int(nodes // 1.2)
    # if nodes < 32:
    #     nodes = 32
    for i in range(KERAS_LAYERS):
        model.add(Dense(nodes,
                        activation='relu', kernel_constraint=keras.constraints.maxnorm(KERAS_MAXNORM)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout, seed=random_state))
        dropout = dropout - 0.1
        if dropout < 0.1:
            dropout = 0.1
        nodes = int(nodes // 2)
        if nodes < 32:
            nodes = 32
    model.add(Dense(2, activation='softmax'))
    optimizer = Adam(lr=KERAS_LEARNING_RATE, decay=decay)
    # optimizer = Adam(lr=KERAS_LEARNING_RATE)

    # Use Early-Stopping
    callback_early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_keras_auc', patience=KERAS_EARLY_STOPPING, verbose=VERBOSE, mode='max')
    callback_tensorboard = keras.callbacks.TensorBoard(
        log_dir=DATA_DIR + '/tensorboard', histogram_freq=1, batch_size=KERAS_BATCH_SIZE,
        write_graph=True, write_grads=True, write_images=False)
    callback_checkpoint = keras.callbacks.ModelCheckpoint(
        model_weight_path, monitor='val_keras_auc', verbose=VERBOSE, save_best_only=True, mode='max')
    callback_gini_metric = keras_gini(
        validation_data=(X_train, Y_train), patience=KERAS_EARLY_STOPPING, save_best=True, verbose=VERBOSE)

    # Compile model
    # model.compile(loss='binary_crossentropy',
    #              optimizer=optimizer, metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=[keras_auc])

    model.summary()

    logger.info("Training ...")
    start = time.time()
    # Training model
    history = model.fit(X_train, Y_train,
                        validation_split=KERAS_VALIDATION_SPLIT,
                        batch_size=KERAS_BATCH_SIZE,
                        epochs=KERAS_N_ROUNDS,
                        callbacks=[
                            # callback_tensorboard,
                            callback_early_stopping,
                            callback_checkpoint,
                        ],
                        class_weight=class_weight_dict,
                        verbose=VERBOSE
                        )
    end = time.time() - start
    # logger.debug("Best score:" + str(callback_gini_metric.best) +
    #              ". Epoch: " + str(callback_gini_metric.best_epoch))
    logger.debug("Train time:" + str(end))
    # load best model
    logger.info("Loading best model ...")
    # if callback_gini_metric.save_best & callback_gini_metric.best_epoch > 0:
    #     model.load_weights(model_weight_path)
    model.load_weights(model_weight_path)

    # serialize model to JSON
    model_json = model.to_json()
    with open(model_path, "w") as json_file:
        json_file.write(model_json)

    logger.debug("Evaluating model ...")
    y_pred = model.predict(X_train, batch_size=KERAS_PREDICT_BATCH_SIZE)
    score = gini_normalized(Y_train[:, 1], y_pred[:, 1])
    logger.debug('Gini:' + str(score))
    logger.debug('Best metric:' + str(callback_early_stopping.best))
    # if callback_early_stopping.stopped_epoch == KERAS_N_ROUNDS-1:
    #     best_round = KERAS_N_ROUNDS
    # else:
    #     best_round = KERAS_N_ROUNDS
    logger.debug(
        'Best round:' + str(callback_early_stopping.stopped_epoch - KERAS_EARLY_STOPPING))

    # Predict and save submission
    logger.info("Predicting and saving result ...")
    # data = eval_data.drop('id', axis=1)
    data = eval_data
    X_eval = scaler.transform(data)
    Y_eval = model.predict(X_eval, batch_size=KERAS_PREDICT_BATCH_SIZE)
    Y_eval = np.absolute(Y_eval)

    eval_output = pd.DataFrame({'id': eval_id, label: Y_eval[:, 1]})
    print(len(eval_output))
    today = str(dtime.date.today())
    logger.debug("Date:" + today)
    eval_output.to_csv(
        DATA_DIR + '/' + today + '-submission.csv.gz', index=False, float_format='%.5f',
        compression='gzip')

    # summarize history for accuracy
    # https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
    # logger.debug("metric:" + str(history.history))
    plt.figure(figsize=(20, 15))
    # plt.subplot(3, 1, 1)
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # summarize history for loss
    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # Summmary auc score
    plt.subplot(2, 1, 2)
    plt.plot(history.history['keras_auc'])
    plt.plot(history.history['val_keras_auc'])
    plt.title('model auc')
    plt.ylabel('auc')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # Summmary gini score
    # plt.subplot(4, 1, 4)
    # plt.plot(callback_gini_metric.history['gini'])
    # plt.title('gini score')
    # plt.ylabel('score')
    # plt.xlabel('epoch')
    # plt.legend(['gini', loc='upper left')
    plt.savefig(DATA_DIR + "/history.png")
    logger.info("Done!")
