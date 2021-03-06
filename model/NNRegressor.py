# @Author: latuannetnam@gmail.com
# class NNRegression: Using Neural Network to predict numerical values
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as ax
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score, train_test_split
from math import sqrt
import io
import time
import os
# from sklearn.preprocessing import StandardScaler, MinMaxScaler


class NNRegressor:
    np.set_printoptions(precision=3)
    np.set_printoptions(suppress=True)
    LOG_ROOT_DIR = "/tmp/tensorflow_logs/"
    NUM_THREADS = 8

    def __init__(self,
                 n_layers=5, n_neurals=50,
                 learning_rate=0.1,
                 epochs=100, batch_size=100,
                 split_ratio=1
                 ):
        # self.sess = tf.Session(config=tf.ConfigProto(
        #     intra_op_parallelism_threads=NNRegressor.NUM_THREADS))
        self.sess = tf.Session()
        self.logs_path = NNRegressor.LOG_ROOT_DIR + "nnregressor"
        # number of neurals per hidden layer1, if n_neurals<2: use simple linear
        # regression
        self.n_neurals = n_neurals
        self.n_layers = n_layers  # numer of hidden layers
        self.learning_rate = learning_rate  # learning rate
        self.split_ratio = split_ratio  # split ratio between training data and test data
        self.epochs = epochs  # total number of training loops
        self.batch_size = batch_size  # batch size for training
        # preload input into queue
        # self.preload()

    def __del__(self):
        self.coord.request_stop()
        self.coord.join(self.threads)
        self.sess.close()

    def preload(self):  # preload data into queue
        with tf.name_scope('Input'):
            # Input data
            self.train_x_initializer = tf.placeholder(
                self.train_X.dtype,
                shape=(self.train_X.shape[0], self.n_features))
            self.train_y_initializer = tf.placeholder(
                self.train_Y.dtype,
                shape=(self.train_Y.shape[0], 1))
            self.batch_train_X = tf.Variable(
                self.train_x_initializer, trainable=False, collections=[], name='Input_X')
            self.batch_train_Y = tf.Variable(
                self.train_y_initializer, trainable=False, collections=[], name='Label_Y')

            train_input, train_label = tf.train.slice_input_producer(
                [self.batch_train_X, self.batch_train_Y], num_epochs=self.epochs)

            self.X, self.Y = tf.train.batch(
                [train_input, train_label],
                batch_size=self.batch_size,
                allow_smaller_final_batch=True)

    def preload_nobatch(self):
        with tf.variable_scope('Input_X', reuse=False):  # place holder for Input
            self.X = tf.placeholder(self.train_X.dtype, shape=(
                None, self.n_features), name="X")
        with tf.variable_scope('Label_Y', reuse=False):  # place holder for Label
            self.Y = tf.placeholder(self.train_Y.dtype, shape=(None, 1), name='Y')

    def dump_input(self):
        # train = np.c_[self.train_X, self.train_Y]
        print("Train X size:", self.train_X.shape,
              " Label size:", self.train_Y.shape)
        test = np.c_[self.test_X, self.test_Y]
        print("test size:", test.shape)
        print("Number of features:", self.n_features)
        print("Number of hidden layers:",
              str(self.n_layers) + " neurons:" + str(self.n_neurals))
        print('Epochs:' + str(self.epochs) + " batch:" +
              str(self.batch_size) +
              " alpha:" + str(self.learning_rate))
        print("Number of steps:" + str(self.epochs * self.train_X.shape[0] /
                                       self.batch_size))

    def inference(self, X, reuse=False):   # define neural network
        if self.n_layers == 0:  # use linear regression
            with tf.variable_scope('Model', reuse=reuse):
                # model = tf.add(tf.matmul(X, self.W), self.b)
                model = tf.layers.dense(
                    X, 1)
        else:  # use neural network
            with tf.variable_scope('Neural_Net', reuse=reuse):
                if self.n_layers == 1:
                    activation = tf.nn.tanh
                else:
                    activation = tf.nn.relu
                initializer = tf.contrib.layers.xavier_initializer()
                # regularizer = tf.contrib.layers.l2_regularizer(0.5)
                regularizer = None
                # initializer = None
                last_layer = X
                n_neurals = self.n_neurals
                for n_layer in range(1, self.n_layers + 1):
                    if reuse==False:
                        print("Hidden Layer:", n_layer, " n_neurals:", n_neurals)
                    layer = tf.layers.dense(
                        last_layer, n_neurals, activation=activation,
                        kernel_initializer=initializer,
                        bias_initializer=initializer,
                        kernel_regularizer=regularizer,
                        bias_regularizer=regularizer,
                        activity_regularizer=regularizer,
                        name="Hidden-" + str(n_layer))
                    last_layer = layer
                    n_neurals = int(n_neurals / 2)
                    if n_neurals < 2:
                        n_neurals = 2

                model = tf.layers.dense(
                    last_layer, 1, name="Output",
                    kernel_initializer=initializer,
                    bias_initializer=initializer,
                    kernel_regularizer=regularizer,
                    bias_regularizer=regularizer,
                    activity_regularizer=regularizer,
                )
        return model

    def loss(self, X, Y, reuse=False):
        Y_predicted = self.inference(X)
        with tf.variable_scope("Loss", reuse=reuse):
            # cost = tf.reduce_sum(tf.squared_difference(
            #     Y, Y_predicted))
            # cost = tf.nn.l2_loss(Y_predicted - Y)
            cost = tf.sqrt(tf.reduce_mean(tf.squared_difference(Y, Y_predicted)))
            
        return cost

    def rmse(self, y_true, y_prediction):
        return sqrt(mean_squared_error(y_true=y_true, y_pred=y_prediction))

    def train(self, total_loss):
        with tf.variable_scope('Train', reuse=False):
            optimizer = tf.train.AdamOptimizer(
                self.learning_rate).minimize(total_loss)
        return optimizer

    def plot(self, X, Y, show=True):
        # Graphic display
        predicted = self.sess.run(self.predict_model(X))
        title = str(self.step) + ': Epochs:' + str(self.epochs) + \
            " batch:" + str(self.batch_size) + \
            " layers:" + str(self.n_layers) + \
            " neurons:" + str(self.n_neurals) + \
            " alpha:" + str(self.learning_rate)
        if self.n_features == 1:  # plot 2D image
            plt.title(title)
            # plt.plot(self.train_X, self.train_Y, c='b', label='Original data')
            # plt.plot(self.train_X, predicted, c='r', label='Fitted')
            plt.scatter(X, Y,
                        c='b', label='Original data')
            plt.scatter(X, predicted, c='r', label='Fitted')

            plt.legend()
        else:  # plot 3D for X1, X2
            if (self.n_features == 2):
                train_X1, train_X2 = np.hsplit(X, self.n_features)
            else:
                # train_X1, train_X2, _ = np.hsplit(
                #     X, self.n_features)
                train_X1 = X[:, 0]
                train_X2 = X[:, 1]
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.text2D(0.05, 0.95, title, transform=ax.transAxes)
            # line plot
            # x = np.squeeze(np.asarray(train_X1))
            # y = np.squeeze(np.asarray(train_X2))
            # z = np.squeeze(np.asarray(Y))
            # z1 = np.squeeze(np.asarray(predicted))
            # ax.plot(x, y, zs=z, c='b', label='Original data')
            # ax.plot(x, y, zs=z1, c='r', label='Fitted')

            # statter plot
            ax.scatter(train_X1, train_X2, Y, c='b',
                       marker='s', label='Original data')
            ax.scatter(train_X1, train_X2, predicted,
                       c='r', marker='v', label='Fitted')

            # trisurf plot
            # ax.plot_trisurf(np.ravel(train_X1),
            #                 np.ravel(train_X2), np.ravel(Y), color='b')
            # print(predicted.shape, np.ravel(predicted).shape)
            # ax.plot_trisurf(np.ravel(train_X1),
            # np.ravel(train_X2), np.ravel(predicted), color='r')

            ax.set_xlabel('X1')
            ax.set_ylabel('X2')
            ax.set_zlabel('Y')
            ax.legend()

        # save image to buffer
        buf = io.BytesIO()
        if show:
            # If want to view image in Tensorboard, do not show plot => Strange
            # bug!!!
            image_path = self.logs_path + "/images"
            if not os.path.exists(image_path):
                os.mkdir(image_path)
            filename = image_path + "/" + \
                time.strftime("%Y%m%d-%H%M%S", time.localtime()) + ".png"
            plt.savefig(filename, format='png')
            plt.show()
        else:
            plt.savefig(buf, format='png')
            buf.seek(0)
        plt.close()
        # plt.clf()
        return buf

    def save_image(self, X, Y):
        plot_buf = self.plot(X, Y, show=False)
        # plot_buf = plot()
        image = tf.image.decode_png(plot_buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        # Add image summary
        image_summary_op = tf.summary.image(
            "model-" + str(self.step), image)
        image_summary = self.sess.run(image_summary_op)
        self.summary_writer.add_summary(image_summary)

    def predict_model(self, X):
        Y_predicted = self.inference(
            tf.convert_to_tensor(X), reuse=True)
        return Y_predicted

    def predict(self, X, convert=False):
        if convert:
            X = X.values.astype(float)
        # print("data to predict")
        # print(X[:5])
        return self.sess.run(self.predict_model(X))

    def train_model(self):  # run loop to train model
        total_loss = self.loss(self.X, self.Y)
        # Create a summary to monitor cost tensor
        tf.summary.scalar("loss", total_loss)
        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()
        train_op = self.train(total_loss), total_loss, merged_summary_op

        # Launch the graph in a session, setup boilerplate
        init = [tf.global_variables_initializer(
        ), tf.local_variables_initializer()]
        self.sess.run(init)
        self.summary_writer = tf.summary.FileWriter(
            self.logs_path, graph=tf.get_default_graph())
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(
            sess=self.sess, coord=self.coord)
        self.sess.run(self.batch_train_X.initializer,
                      feed_dict={self.train_x_initializer: self.train_X})
        self.sess.run(self.batch_train_Y.initializer,
                      feed_dict={self.train_y_initializer: self.train_Y})
        # actual training loop
        try:
            train_loss = 0
            print_step = (self.epochs * self.train_X.shape[0] /
                          self.batch_size) // 10
            if print_step == 0:
                print_step = 1
            print("Print step:" + str(print_step))
            self.step = 0
            while not self.coord.should_stop():
                # Run one step of the model.
                result = self.sess.run([train_op])
                train_loss = result[0][1]
                summary = result[0][2]
                # Write logs at every iteration
                self.summary_writer.add_summary(summary, self.step + 1)
                if self.step % print_step == 0:
                    print("step:", self.step + 1, " train loss: ", train_loss)
                    # self.save_image(self.train_X, self.train_Y)
                self.step += 1
        except tf.errors.OutOfRangeError:
            print('Done training for:', self.step, " epochs")
        finally:
            # When done, ask the threads to stop.
            # print(" final loss:", train_loss, " test lost:", test_loss)
            # self.save_image(self.input_X, self.label_Y)
            # self.plot(self.train_X, self.train_Y)
            y_pred = self.predict(self.test_X)
            print("Y_predict of train data")
            print(y_pred[:5])
            print("RMSE:", self.rmse(self.test_Y, y_pred))

    def train_model_nobatch(self):
        total_loss = self.loss(self.X, self.Y)
        # Create a summary to monitor cost tensor
        tf.summary.scalar("loss", total_loss)
        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()
        train_op = self.train(total_loss), total_loss, merged_summary_op

        # Launch the graph in a session, setup boilerplate
        init = [tf.global_variables_initializer(
        ), tf.local_variables_initializer()]
        self.sess.run(init)
        self.summary_writer = tf.summary.FileWriter(
            self.logs_path, graph=tf.get_default_graph())
        self.coord = tf.train.Coordinator()
        self.threads = tf.train.start_queue_runners(
            sess=self.sess, coord=self.coord)
        print_step = self.epochs // 10
        if print_step == 0:
            print_step = 1
        for self.step in range(self.epochs):
            result = self.sess.run([train_op], feed_dict={
                self.X: self.train_X, self.Y: self.train_Y})
            train_loss = result[0][1]
            summary = result[0][2]
            # Write logs at every iteration
            self.summary_writer.add_summary(summary, self.step + 1)
            if self.step % print_step == 0:
                print("step:", self.step + 1, " train loss: ", train_loss)

        y_pred = self.predict(self.test_X)
        print("Y_predict of train data")
        print(y_pred[:5])
        print("RMSE:", self.rmse(self.test_Y, y_pred))

    def fit(self, X, Y, convert=False):
        if convert:
            input_X = X.values.astype(float)
            label_Y = np.reshape(Y.values.astype(float), (-1, 1))
        else:
            input_X = X.astype(float)
            label_Y = np.reshape(Y.astype(float), (-1, 1))
        sample_size = input_X.shape[0]
        # size of training data
        self.train_size = int(sample_size * self.split_ratio)

        if self.batch_size >= self.train_size:
            self.batch_size = self.train_size
        # self.train_X = input_X[:self.train_size]  # train data
        # self.test_X = input_X[self.train_size:]   # test data
        # self.train_Y = label_Y[:self.train_size]  # train label
        # self.test_Y = label_Y[self.train_size:]   # test label
        self.train_X, self.test_X, self.train_Y, self.test_Y = train_test_split(
            input_X, label_Y, train_size=self.train_size, random_state=324)
        # number of features of train data: X1, X2 ...
        self.n_features = input_X.shape[1]
        self.dump_input()
        # self.preload()
        self.preload_nobatch()
        self.train_model_nobatch()
