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

# System
import datetime as dtime
import time

pd.options.display.float_format = '{:,.4f}'.format
# Input data files are available in the DATA_DIR directory.
DATA_DIR = "data-temp"

# Define the gini metric - from https://www.kaggle.com/c/ClaimPredictionChallenge/discussion/703#5897


def gini2(actual, pred, cmpcol=0, sortcol=1):
    assert(len(actual) == len(pred))
    all = np.asarray(
        np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)


def gini2_normalized(a, p):
    return gini2(a, p) / gini2(a, a)


class keras_gini(keras.callbacks.Callback):
    def __init__(self, validation_data, classifier=True):
        #         print("init validation len:", len(validation_data))
        self.validation_data = validation_data
        self.classifier = classifier
        self.maps = []

    def eval_metric(self):
        # print("")
        # print("validation len:", len(self.validation_data))
        # print(self.validation_data)
        x_val = self.validation_data[0]
        y_true = self.validation_data[1]
        y_pred = self.model.predict(x_val)
        # print("Y_pred:", y_pred[:5])
        if self.classifier:
            score = gini2_normalized(y_true[:, 1], y_pred[:, 1])
        else:
            score = gini2_normalized(y_true, y_pred)
        return score

    def on_epoch_end(self, epoch, logs={}):
        score = self.eval_metric()
        print(". Eval for epoch %d is %f" % (epoch + 1, score))
        self.maps.append(score)


# Load data. Download from:https://www.kaggle.com/c/nyc-taxi-trip-duration/data
train_data = pd.read_csv(DATA_DIR + "/train.csv")
eval_data = pd.read_csv(DATA_DIR + "/test.csv")
print("train size:", train_data.shape, " test size:", eval_data.shape)

label = 'target'
features = eval_data.columns.values
target = train_data[label]
data = train_data.drop(['id', label], axis=1)

# Compute class weigth to balance label
cw = class_weight.compute_class_weight('balanced', np.unique(target), target)
class_weight_dict = dict(enumerate(cw))

# Split train/test set
X_train, X_test, Y_train, Y_test = train_test_split(
    data, target, train_size=0.85, random_state=1234, stratify=target)
print("X_train:", X_train.shape, " Y_train:", Y_train.shape,
      " X_test:", X_test.shape, " Y_test:", Y_test.shape)

# Scaling features
scaler = StandardScaler()
scaler.fit(X_train)
# Transform train data
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Transform label to categorial
Y_train = np_utils.to_categorical(Y_train.values)
Y_test = np_utils.to_categorical(Y_test.values)
print("Y-train:", Y_train[:5])

# Model definition
KERAS_LEARNING_RATE = 0.001
KERAS_N_ROUNDS = 20
KERAS_BATCH_SIZE = 32
KERAS_NODES = 64
KERAS_LAYERS = 5
KERAS_DROPOUT_RATE = 0.2
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
    monitor='val_loss', patience=2, verbose=1, mode='auto')
callback_tensorboard = keras.callbacks.TensorBoard(
    log_dir=DATA_DIR + '/tensorboard', histogram_freq=1, batch_size=32, write_graph=True, write_grads=True, write_images=True)
callback_gini_metric = keras_gini(
    validation_data=(X_test, Y_test), classifier=True)

# Compile model
model.compile(loss='binary_crossentropy',
              optimizer=optimizer, metrics=['accuracy'])
model.summary()
start = time.time()
# Training model
model.fit(X_train, Y_train,
          validation_data=(X_test, Y_test),
          batch_size=KERAS_BATCH_SIZE,
          epochs=KERAS_N_ROUNDS,
          callbacks=[callback_early_stopping,
                     # callback_tensorboard,
                     callback_gini_metric
                     ],
          class_weight=class_weight_dict,
          verbose=True
          )
end = time.time() - start

print("Train time:", end)
# Evaluate model
score = model.evaluate(X_test, Y_test, verbose=1)
print("")
print("Test score:", score[0])
print('Test accuracy:', score[1])

# Gini
y_pred = model.predict(X_test)
print(y_pred[:5])
score = gini2_normalized(Y_test[:, 1], y_pred[:, 1])
# score = gini_normalized(Y_test, y_pred)
print('Score:', score)

# Predict and save submission
data = eval_data.drop('id', axis=1)
# data = eval_set[real_vars]
X_eval = scaler.transform(data)
Y_eval = model.predict(X_eval)
Y_eval = np.absolute(Y_eval)

eval_output = pd.DataFrame({'id': eval_data['id'], label: Y_eval[:, 1]})
print(len(eval_output))
today = str(dtime.date.today())
print(today)
eval_output.to_csv(
    DATA_DIR + '/' + today + '-submission.csv.gz', index=False, float_format='%.5f',
    compression='gzip')

print("Done!")
