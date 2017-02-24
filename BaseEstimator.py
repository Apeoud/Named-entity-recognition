import tensorflow as tf
import numpy as np
from conllner import Word_Window_DataSet, extract_sentences_labels, read_data_set, DataSet
from load import precision_score, recall_score, get_input, get_sentences
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

        """" initialize the DNN network """
        self._hidden_units = hidden_units
        self.n_input = n_input
        self.n_classes = n_classes
        self._learning_rate = learning_rate

        self.graph = tf.get_default_graph()

        with self.graph.as_default():
            self.weights = dict()
            self.biases = dict()
            self.layers = dict()

            self.x = tf.placeholder('float', [None, self.n_input], name='input_x')
            self.y = tf.placeholder('float', [None, self.n_classes], name='input_y')
            self.prediction = self._construct_net()

            self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.prediction, self.y))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(self.cost)

            self.init = tf.global_variables_initializer()

        self.model_dir = model_dir
        self.batch_size = batch_size

        self._train_epochs = train_epochs
        self._stop_step = stop_step
        self._saver = tf.train.Saver()
        self._sess = tf.Session()

    def _full_connected_layer(self, in_op, input_unit, output_unit, layer_index, out_layer=False):

        if layer_index >= len(self._hidden_units):
            raise ValueError()

        if out_layer:
            layer_name = 'out'
        else:
            layer_name = 'hidden_' + str(layer_index)

        weight = tf.Variable(
            tf.random_uniform([input_unit, output_unit], minval=- math.sqrt(6) / math.sqrt(input_unit + output_unit),
                              maxval=math.sqrt(6) / math.sqrt(input_unit + output_unit)),
            name='w_' + layer_name)
        bias = tf.Variable(tf.random_normal([output_unit]), name='b_' + layer_name)

        layer = tf.add(tf.matmul(in_op, weight), bias)
        if not out_layer:
            layer = tf.nn.relu(layer)

        self.weights[layer_name] = weight
        self.biases[layer_name] = bias
        self.layers[layer_name] = layer

        return layer

    def _construct_net(self):

        n_layers = len(self._hidden_units)

        # input layer
        out_op = self._full_connected_layer(self.x, self.n_input, self._hidden_units[0], 0)

        # hidden layer
        for i in range(1, len(self._hidden_units)):
            out_op = self._full_connected_layer(out_op, self._hidden_units[i - 1], self._hidden_units[i], i)

        # out layer
        out_layer = self._full_connected_layer(out_op, self._hidden_units[-1], self.n_classes, i, out_layer=True)
        return out_layer

    def mlp_ff(self, x, weights, biases):
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['h1'])
        layer_1 = tf.nn.relu(layer_1)

        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['h2'])
        layer_2 = tf.nn.relu(layer_2)

        layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['h3'])
        layer_3 = tf.nn.sigmoid(layer_3)

        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']

        # out_layer = tf.nn.softmax(out_layer)
        return out_layer

    def _construct_graph(self):
        """ construct the graph mainly variables

        Arg:

        """
        x = tf.placeholder('float', [None, self.n_input])
        y = tf.placeholder('float', [None, self.n_classes])

        # weights
        weights = {
            'h1': tf.Variable(
                tf.random_uniform([self.n_input, 300], minval=- math.sqrt(6) / 40, maxval=math.sqrt(6) / 40),
                name='w_h1'),
            'h2': tf.Variable(
                tf.random_uniform([300, 300], minval=- math.sqrt(6) / 40, maxval=math.sqrt(6) / 40),
                name='w_h2'),
            'h3': tf.Variable(
                tf.random_uniform([300, 300], minval=- math.sqrt(6) / 40, maxval=math.sqrt(6) / 40),
                name='w_h3'),
            'out': tf.Variable(
                tf.random_uniform([300, 5], minval=- math.sqrt(6) / 40, maxval=math.sqrt(6) / 40),
                name='w_out')
        }

        biases = {
            'h1': tf.Variable(tf.random_normal([300]), name='b_h1'),
            'h2': tf.Variable(tf.random_normal([300]), name='b_h2'),
            'h3': tf.Variable(tf.random_normal([300]), name='b_h3'),
            'out': tf.Variable(tf.random_normal([self.n_classes]), name='b_out')
        }

        # tf.add_to_collection('vars', weights)
        # tf.add_to_collection('vars', biases)
        #
        # saver = tf.train.Saver()

        pred = self.mlp_ff(x, weights, biases)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
        optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate).minimize(cost)

        # prediction
        y_p = tf.argmax(pred, 1)

        init = tf.global_variables_initializer()

        init = tf.global_variables_initializer()
        self._sess.run(init)

        return x, y, pred, cost, optimizer

    def fit(self, dataset, Y, display_step=50):

        sentences = get_sentences('./data/eng.train')

        sess = tf.Session(graph=self.graph)
        sess.run(self.init)

        for epoch in range(self._train_epochs):
            avg_cost = 0

            total_batch = math.ceil(dataset.num_examples / self.batch_size)

            batch_list = [index for index in range(total_batch)]
            random.shuffle(batch_list)

            for batch_i in batch_list:
                train_x, train_y = get_input(sentences[batch_i * self.batch_size:(batch_i + 1) * self.batch_size])
                # train_x, train_y = dataset.sent2features(
                #     dataset.tokens[batch_i * self.batch_size:(batch_i + 1) * self.batch_size],
                #     dataset.labels[batch_i * self.batch_size:(batch_i + 1) * self.batch_size])
                _, c = sess.run([self.optimizer, self.cost], feed_dict={self.x: train_x, self.y: train_y})
                avg_cost += c / total_batch

            print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                  "{:.9f}".format(avg_cost))

        self._saver.save(sess, self.model_dir)

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

    def precision_score(self, y_pred, y_true):
        """
        Arg:
            y_pred : prediction of of the test data, one-hot format
            y_true : label of the test data , one-hot format

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

        # perm = [i for i in range(len(y_true)) if y_true[i] != 0]
        # real_pred = y_pred[perm]
        # real_true = y_true[perm]
        #
        # acc = np.mean(np.equal(real_pred, real_true))
        #
        # return acc

    def evaluate(self, test_x, test_y):

        sess = tf.Session(graph=self.graph)
        self._saver.restore(sess, self.model_dir)

        y_pred = sess.run(self.prediction, feed_dict={self.x: test_x})
        y_pred = np.argmax(y_pred, 1)
        y_true = np.argmax(test_y, 1)

        print('precision: ', precision_score(y_pred, y_true))
        print('recall   : ', recall_score(y_pred, y_true))

        # return y_pred, y_true

        return

    def predict(self, X):
        """ predict the label one_hot = False
            0 : 'O'
            1 : 'PER'
            2 : 'LOC'
            3 : 'ORG'
            4 : 'MISC'
        """
        return np.argmax(self.proba(X), 1)

    def proba(self, X):
        """ predict the unlabeled data
        Arg:
            X : to be predicted , (n_examples, n_features)

        Return:
            y_pred : predicted label, one hot (n_examples, n_classes)
        """

        sess = tf.Session(graph=self.graph)

        self._saver.restore(sess, self.model_dir)
        y_pred = sess.run(self.prediction, feed_dict={self.x: X})

        return y_pred


if __name__ == "__main__":
    train = read_data_set('w2v')

    mlp = MultiLayerPerceptron(
        hidden_units=[300, 300],
        train_epochs=10,
        n_input=2100,
        n_classes=5,
        batch_size=100,
        learning_rate=0.005,
        model_dir='./tmp/models/tf_mlp.ckpt'
    )

    test = conllner.read_test_data_set('w2v')
    test_x = test._extract_feature(test._data)
    test_y = test._extract_label(test._data)

    mlp.fit(train, [], display_step=50)

    # test_x, test_y = get_input(get_sentences('./data/eng.testb'))
    mlp.evaluate(test_x, test_y)
    print('end')
