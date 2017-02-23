import tensorflow as tf
import numpy as np
from conllner import Word_Window_DataSet, extract_sentences_labels, read_data_set
import conllner
import math
import random


class BaseEstimator(object):
    def __init__(self):
        return

    def fit(self, X, y):
        return

    def predict(self, X):
        return

    def save(self):
        return

    def load(self):
        return


class MultiLayerPerceptron(BaseEstimator):
    def __init__(self,
                 hidden_units,
                 n_input,
                 n_classes,
                 model_dir,
                 stop_step=100,
                 train_epochs=1000,
                 learning_rate=0.01,
                 batch_size=200):
        self._hidden_units = hidden_units
        self.n_input = n_input
        self.n_classes = n_classes
        self.model_dir = model_dir
        self.batch_size = batch_size
        self._learning_rate = learning_rate
        self._train_epochs = train_epochs
        self._stop_step = stop_step
        self._saver = None
        self._sess = tf.Session()

    def precision_score(self, y_pred, y_true):
        """

        :param pred:
        :param y:
        :return:
        """
        y_true = np.argmax(y_true, 1)
        y_pred = np.argmax(y_pred, 1)

        tp = [0] * 5
        p = [0] * 5
        for i in range(len(y_pred)):
            if y_true[i] == y_pred[i]:
                tp[y_pred[i]] += 1
            p[y_pred[i]] += 1

        return np.sum(tp[1:] / np.sum(p[1:]))

        #
        #
        # perm = [i for i in range(len(y_true)) if y_true[i] != 0]
        # real_pred = y_pred[perm]
        # real_true = y_true[perm]
        #
        # acc = np.mean(np.equal(real_pred, real_true))
        #
        # return acc

    def construct_variables(self):
        """ construct tf variables
        """
        if self._hidden_units == []:
            raise ValueError("invalid unit parameters")
        n_layers = len(self._hidden_units)

        return

    def _construct_graph(self):
        """ construct the graph mainly variables

        Arg:

        """
        x = tf.placeholder('float', [None, self.n_input])
        y = tf.placeholder('float', [None, self.n_classes])

        weights = []
        biases = []

        # weights and biases
        weights = dict()
        biases = dict()
        weights['h1'] = tf.Variable(
            tf.random_uniform([self.n_input, self._hidden_units[0]], minval=- math.sqrt(6) / 40,
                              maxval=math.sqrt(6) / 40),
            name='w_h1')
        biases['h1'] = tf.Variable(tf.random_normal([self._hidden_units[0]]), name='b_h1')
        weights['out'] = tf.Variable(
            tf.random_uniform([self._hidden_units[-1], self.n_classes], minval=- math.sqrt(6) / 40,
                              maxval=math.sqrt(6) / 40),
            name='w_out'
        )
        biases['out'] = tf.Variable(tf.random_normal([self.n_classes]), name='b_out')
        for i in range(len(self._hidden_units) - 1):
            weights['h' + str(i + 2)] = tf.Variable(
                tf.random_uniform([self._hidden_units[i], self._hidden_units[i + 1]], minval=- math.sqrt(6) / 40,
                                  maxval=math.sqrt(6) / 40),
                name='w_h' + str(i + 2)
            )
            biases['h' + str(i + 2)] = tf.Variable(tf.random_normal([self._hidden_units[i + 1]]),
                                                   name='b_h' + str(i + 2))

        # weights and biases

        def mlp_ff(x, weights, biases):
            layer_in = tf.add(tf.matmul(x, weights['h1']), biases['h1'])
            layer_out = tf.nn.relu(layer_in)
            for i in range(len(self._hidden_units) - 1):
                layer_out = tf.add(tf.matmul(layer_in, weights['h' + str(i + 2)]), biases['h' + str(i + 2)])
                layer_out = tf.nn.relu(layer_out)
                layer_in = layer_out

            out_layer = tf.matmul(layer_out, weights['out']) + biases['out']
            # out_layer = tf.nn.softmax(out_layer)
            return out_layer

        pred = mlp_ff(x, weights, biases)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(cost)

        init = tf.global_variables_initializer()
        self._sess.run(init)

        return x, y, pred, cost, optimizer

    def fit(self, dataset, test_set, display_step=50):

        x, y, pred, cost, optimizer = self._construct_graph()

        # define evaluate
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        self._saver = tf.train.Saver()

        step = 0
        for epoch in range(self._train_epochs):
            avg_cost = 0

            total_batch = math.ceil(dataset.num_examples / self.batch_size)

            batch_list = [index for index in range(total_batch)]
            random.shuffle(batch_list)

            for batch_i in batch_list:
                train_x, train_y = dataset.sent2features(
                    dataset.tokens[batch_i * self.batch_size:(batch_i + 1) * self.batch_size],
                    dataset.labels[batch_i * self.batch_size:(batch_i + 1) * self.batch_size])
                _, c = self._sess.run([optimizer, cost], feed_dict={x: train_x, y: train_y})
                avg_cost += c / total_batch

            print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                  "{:.9f}".format(avg_cost))

            # opt, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            # if dataset.epochs_completed - start_epoch == 1:
            #     y_pred = sess.run(pred, feed_dict={x: batch_x, y: batch_y})
            #     acc = self.precision_score(y_pred, batch_y)
            #     print("Iter " + str(step) + ", Mini-batch accuracy = " + \
            #           "{:.6f}".format(acc) + ", loss = " + "{:.5f}".format(c))
            #
            #     test_x, test_y = test_set.sent2features(test_set.tokens, test_set.labels)
            #
            #     y_pred = sess.run(pred, feed_dict={x: test_x, y: test_y})
            #     # y_true = np.argmax(test_y, 1)
            #
            #     print('precision: ', self.precision_score(y_pred, test_y))

        self._saver.save(self._sess, self.model_dir)

        return

    def save(self, model_path):
        self._saver.save(self._sess, model_path)
        self.model_dir = model_path
        return self.model_dir

    def load(self, model_path):
        """ before load, you must initialize the session
        """
        self._saver = tf.train.Saver()
        self._saver.restore(self._sess, self._hidden_units)
        return

    def evaluate(self, X, y):

        x, y, pred, cost, optimizer = self._construct_graph()
        self._saver = tf.train.Saver()
        self._saver.restore(self._sess, self.model_dir)

        with self._sess as sess:
            y_pred = sess.run(pred, feed_dict={x: test_x, y: test_y})
            # y_pred = np.argmax(y_pred, 1)
            # y_true = np.argmax(test_y, 1)

            print('precision: ', self.precision_score(y_pred, test_y))

            # return y_pred, y_true

        return

    def predict(self, X):
        with tf.Session() as sess:
            self._saver.restore(sess, self.model_dir)
            y_pred = sess.run()

        return y_pred




if __name__ == "__main__":

    train = read_data_set('w2v')

    mlp = MultiLayerPerceptron(
        hidden_units=[300, 300],
        train_epochs=5,
        n_input=2100,
        n_classes=5,
        batch_size=100,
        learning_rate=0.005,
        model_dir='./tmp/models/mlp.ckpt'
    )

    conll_test = conllner.read_test_data_set('w2v')
    test_x, test_y = conll_test.sent2features(conll_test.tokens, conll_test.labels)

    mlp.fit(train, conll_test, display_step=50)
    mlp.evaluate(test_x, test_y)
