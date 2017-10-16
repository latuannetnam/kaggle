# System
import datetime as dtime
import time
import logging
import sys

# data processing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ML
# # Scikit-learn
from sklearn.model_selection import cross_val_score, train_test_split, learning_curve, validation_curve, KFold
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelBinarizer, StandardScaler

# LightGBM
from lightgbm import LGBMClassifier, LGBMRegressor

# Keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.models import model_from_json


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
        y_pred = self.model.predict(x_val)
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
    # Load data. Download from:https://www.kaggle.com/c/nyc-taxi-trip-duration/data
    logger.info("Loading data ...")
    train_data = pd.read_csv(DATA_DIR + "/train.csv")
    eval_data = pd.read_csv(DATA_DIR + "/test.csv")
    logger.debug("train size:" + str(train_data.shape) + " test size:" + str(eval_data.shape))

    label = 'target'
    features = eval_data.columns.values
    target = train_data[label]
    data = train_data.drop(['id', label], axis=1)

    # Compute class weigth to balance label
    cw = class_weight.compute_class_weight(
        'balanced', np.unique(target), target)
    class_weight_dict = dict(enumerate(cw))

    # Split train/test set
    # X_train, X_test, Y_train, Y_test = train_test_split(
    #     data, target, train_size=0.85, random_state=1234, stratify=target)
    X_train = data
    Y_train = target
    logger.debug("X_train:" + str(X_train.shape) + " Y_train:" + str(Y_train.shape))
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
    KERAS_N_ROUNDS = 20
    KERAS_BATCH_SIZE = 32
    KERAS_NODES = 64
    KERAS_LAYERS = 10
    KERAS_DROPOUT_RATE = 0.2
    VALIDATION_SPLIT = 0.2
    VERBOSE = True
    model_weight_path = DATA_DIR + "/model_weight.h5"
    model_path = DATA_DIR + "/model.json"
    random_state = 12343
    # n_features = len(data.columns) - 2
    n_features = len(data.columns)
    decay = KERAS_LEARNING_RATE / KERAS_N_ROUNDS
    # create model
    model = Sequential()
    model.add(Dense(KERAS_NODES, input_shape=(n_features, ),
                    activation='relu'))
    model.add(Dropout(KERAS_DROPOUT_RATE, seed=random_state))
    for i in range(KERAS_LAYERS):
        model.add(Dense(KERAS_NODES,
                        activation='relu'))
        model.add(Dropout(KERAS_DROPOUT_RATE, seed=random_state))
    model.add(Dense(2, activation='softmax'))
    optimizer = Adam(lr=KERAS_LEARNING_RATE, decay=decay)
    # optimizer = Adam(lr=KERAS_LEARNING_RATE)

    # Use Early-Stopping
    callback_early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, verbose=VERBOSE, mode='auto')
    callback_tensorboard = keras.callbacks.TensorBoard(
        log_dir=DATA_DIR + '/tensorboard', histogram_freq=1, batch_size=32, write_graph=True, write_grads=True, write_images=True)
    callback_gini_metric = keras_gini(
        validation_data=(X_train, Y_train), patience=20, save_best=True, verbose=VERBOSE)

    # Compile model
    # model.compile(loss='binary_crossentropy',
    #              optimizer=optimizer, metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])

    model.summary()

    logger.info("Training ...")
    start = time.time()
    # Training model
    # model.fit(X_train, Y_train,
    #           validation_data=(X_test, Y_test),
    #           batch_size=KERAS_BATCH_SIZE,
    #           epochs=KERAS_N_ROUNDS,
    #           callbacks=[callback_early_stopping,
    #                      # callback_tensorboard,
    #                      callback_gini_metric
    #                      ],
    #           class_weight=class_weight_dict,
    #           verbose=True
    #           )
    history = model.fit(X_train, Y_train,
                        validation_split=VALIDATION_SPLIT,
                        batch_size=KERAS_BATCH_SIZE,
                        epochs=KERAS_N_ROUNDS,
                        callbacks=[
                            callback_gini_metric
                        ],
                        class_weight=class_weight_dict,
                        verbose=VERBOSE
                        )
    end = time.time() - start
    logger.debug("Best score:" + str(callback_gini_metric.best) +
                 ". Epoch: " + str(callback_gini_metric.best_epoch))
    logger.debug("Train time:" + str(end))

    # load best model
    logger.info("Evaluating model ...")
    if callback_gini_metric.save_best & callback_gini_metric.best_epoch > 0:
        model.load_weights(model_path)

    # serialize model to JSON
    model_json = model.to_json()
    with open(model_path, "w") as json_file:
        json_file.write(model_json)

    # Evaluate model
    score = model.evaluate(X_train, Y_train, verbose=1)
    logger.debug("accuracy:" + str(score[0]) + ". loss:" + str(score[1]))

    # Gini
    y_pred = model.predict(X_train)
    # print(y_pred[:5])
    score = gini_normalized(Y_train[:, 1], y_pred[:, 1])
    logger.debug('Gini score:' + str(score))

    # Predict and save submission
    logger.info("Predicting and saving result ...")
    data = eval_data.drop('id', axis=1)
    # data = eval_set[real_vars]
    X_eval = scaler.transform(data)
    Y_eval = model.predict(X_eval)
    Y_eval = np.absolute(Y_eval)

    eval_output = pd.DataFrame({'id': eval_data['id'], label: Y_eval[:, 1]})
    print(len(eval_output))
    today = str(dtime.date.today())
    logger.debug("Date:" + today)
    eval_output.to_csv(
        DATA_DIR + '/' + today + '-submission.csv.gz', index=False, float_format='%.5f',
        compression='gzip')

    # summarize history for accuracy
    # https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 3, 1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.subplot(1, 3, 2)
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.subplot(1, 3, 3)
    # Summmary gini score
    plt.plot(callback_gini_metric.history['gini'])
    plt.title('gini score')
    plt.ylabel('score')
    plt.xlabel('epoch')
    # plt.legend(['gini', loc='upper left')
    plt.savefig(DATA_DIR + "/history.png")
    logger.info("Done!")
