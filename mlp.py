import numpy as np
import re
import gc
import math
import random
import tensorflow as tf
from load import get_sentences, get_input
from load import precision_score, recall_score
import tflearn
from conllner import read_data_set
import conllner

# parameters
training_epochs = 5
learning_rate = 0.005
batch_size = 100

# Network parameters
n_hidden_1 = 300
n_hidden_2 = 300
n_hidden_3 = 300
n_input = 2100
n_classes = 5

train_path = './data/eng.train'
test_path = './data/eng.testa'


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


def train(training_epoch=5, flag=2):
    # save the model


    # placeholder
    x = tf.placeholder('float', [None, n_input])
    y = tf.placeholder('float', [None, n_classes])

    # weights
    weights = {
        'h1': tf.Variable(
            tf.random_uniform([n_input, n_hidden_1], minval=- math.sqrt(6) / 40, maxval=math.sqrt(6) / 40),
            name='w_h1'),
        'h2': tf.Variable(
            tf.random_uniform([n_hidden_1, n_hidden_2], minval=- math.sqrt(6) / 40, maxval=math.sqrt(6) / 40),
            name='w_h2'),
        'h3': tf.Variable(
            tf.random_uniform([n_hidden_2, n_hidden_3], minval=- math.sqrt(6) / 40, maxval=math.sqrt(6) / 40),
            name='w_h3'),
        'out': tf.Variable(
            tf.random_uniform([n_hidden_3, n_classes], minval=- math.sqrt(6) / 40, maxval=math.sqrt(6) / 40),
            name='w_out')
    }

    biases = {
        'h1': tf.Variable(tf.random_normal([n_hidden_1]), name='b_h1'),
        'h2': tf.Variable(tf.random_normal([n_hidden_2]), name='b_h2'),
        'h3': tf.Variable(tf.random_normal([n_hidden_3]), name='b_h3'),
        'out': tf.Variable(tf.random_normal([n_classes]), name='b_out')
    }

    # tf.add_to_collection('vars', weights)
    # tf.add_to_collection('vars', biases)
    #
    # saver = tf.train.Saver()

    pred = mlp_ff(x, weights, biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # prediction
    y_p = tf.argmax(pred, 1)

    init = tf.global_variables_initializer()

    # all sentences
    sentences = get_sentences(train_path)

    save_path = ''
    saver = tf.train.Saver()

    if flag == 2:

        with tf.Session() as sess:

            sess.run(init)

            for epoch in range(training_epoch):
                avg_cost = 0
                # each batch contains batch_size sentences
                total_batch = math.ceil((len(sentences) / batch_size))

                batch_list = [index for index in range(total_batch)]
                # random.shuffle(batch_list)

                for batch_i in batch_list:
                    train_x, train_y = get_input(sentences[batch_i * batch_size:(batch_i + 1) * batch_size])
                    _, c = sess.run([optimizer, cost], feed_dict={x: train_x, y: train_y})
                    avg_cost += c / total_batch

                print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                      "{:.9f}".format(avg_cost))

            save_path = saver.save(sess, './tmp/models/tf_mlp.ckpt')
            print(save_path)

    else:

        with tf.Session() as sess:
            sess.run(init)

            print(save_path)
            saver.restore(sess, './tmp/models/tf_mlp.ckpt')
            print("Model restored from file: %s" % save_path)

            # test_x, test_y = get_input(get_sentences(test_path))
            conll_test = conllner.read_test_data_set('w2v')
            test_x, test_y = conll_test.sent2features(conll_test.tokens, conll_test.labels)

            y_pred = sess.run(pred, feed_dict={x: test_x, y: test_y})
            y_pred = np.argmax(y_pred, 1)
            y_true = np.argmax(test_y, 1)

            print('precision: ', precision_score(y_pred, y_true))
            print('recall   : ', recall_score(y_pred, y_true))

            return y_pred, y_true


if __name__ == '__main__':
    # train(flag=2)
    train(flag=1)

