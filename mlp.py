import numpy as np
import re
import gc
import math
import random
import tensorflow as tf
from load import get_sentences, get_input
from load import precision_score, recall_score

# parameters
training_epochs = 5
learning_rate = 0.005
batch_size = 100

# Network parameters
n_hidden_1 = 300
n_hidden_2 = 300
n_hidden_3 = 300
n_input = 2100
n_classes = 8

train_path = './data/eng.train'
test_path = './data/eng.testb'


def mlp_ff(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['h1'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['h2'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['h3'])
    layer_3 = tf.nn.sigmoid(layer_3)

    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    # out_layer = tf.nn.softmax(out_layer)
    return out_layer


def train(training_epoch=10):
    x = tf.placeholder('float', [None, n_input])
    y = tf.placeholder('float', [None, n_classes])

    # weights
    weights = {
        'h1': tf.Variable(
            tf.random_uniform([n_input, n_hidden_1], minval=- math.sqrt(6) / 40, maxval=math.sqrt(6) / 40)),
        'h2': tf.Variable(
            tf.random_uniform([n_hidden_1, n_hidden_2], minval=- math.sqrt(6) / 40, maxval=math.sqrt(6) / 40)),
        'h3': tf.Variable(
            tf.random_uniform([n_hidden_2, n_hidden_3], minval=- math.sqrt(6) / 40, maxval=math.sqrt(6) / 40)),
        'out': tf.Variable(
            tf.random_uniform([n_hidden_3, n_classes], minval=- math.sqrt(6) / 40, maxval=math.sqrt(6) / 40))
    }

    biases = {
        'h1': tf.Variable(tf.random_normal([n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_2])),
        'h3': tf.Variable(tf.random_normal([n_hidden_3])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    pred = mlp_ff(x, weights, biases)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # prediction
    y_p = tf.argmax(pred, 1)

    init = tf.global_variables_initializer()

    # all sentences
    sentences = get_sentences(train_path)


    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(training_epoch):
            avg_cost = 0
            # each batch contains batch_size sentences
            total_batch = math.ceil((len(sentences) / batch_size))

            batch_list = [index for index in range(total_batch)]
            random.shuffle(batch_list)

            for batch_i in batch_list:
                train_x, train_y = get_input(sentences[batch_i * batch_size:(batch_i + 1) * batch_size])
                _, c = sess.run([optimizer, cost], feed_dict={x: train_x, y: train_y})
                avg_cost += c / total_batch

            print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                  "{:.9f}".format(avg_cost))

        test_x, test_y = get_input(get_sentences(test_path))
        y_pred = sess.run(y_p, feed_dict={x: test_x, y: test_y})
        y_true = np.argmax(test_y, 1)

        print('precision: ', precision_score(y_pred, y_true))



def main():
    with tf.Session() as sess:
        sess.run()


if __name__ == '__main__':
    train()
