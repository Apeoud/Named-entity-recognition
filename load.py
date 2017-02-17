import numpy as np
import re
import gc
import math
import random
import tensorflow as tf
import sklearn as sk

# parameters
VECTOR_SIZE = 300
WINDOW_SIZE = 2

# parameters
training_epochs = 10
learning_rate = 0.005
batch_size = 100

n_hidden_1 = 300
n_hidden_2 = 300
n_hidden_3 = 300
n_input = 2100
n_classes = 5

data_path = './data'


def get_sentences(filename):
    sentences = []
    sentence = []

    tag = dict()

    with open(filename, 'r') as rFile:
        for line in rFile.readlines():
            arr = line.strip().split()
            if len(arr) != 4:
                sentences.append(sentence)
                sentence = []
            else:
                sentence.append(arr)
                if arr[3] not in tag.keys():
                    tag[arr[3]] = 0
                else:
                    tag[arr[3]] += 1

    return sentences


SENTENCE_SIZE = len(get_sentences(data_path + '/eng.train'))
BLOCK_SIZE = 3000


def get_vec(word):
    rand = np.random.uniform(-0.25, 0.25, VECTOR_SIZE)
    s = re.sub('[^0-9a-zA-Z]+', '', word)
    vec = []
    if word == 'space':
        vec = [0 for _ in range(VECTOR_SIZE)]
    elif word in word_set:
        vec = word_set[word]
    elif word.lower() in word_set:
        vec = word_set[word.lower()]
    elif s in word_set:
        vec = word_set[s]
    elif s.isdigit():
        vec = word_set['1']

    if len(vec) > 0:
        return np.asarray(vec)
    return rand


def get_input(sentences):
    features = []
    tags = []
    # sentences = get_sentences(filename)[sent_begin:sent_end]
    for sentence in sentences:
        for i in range(len(sentence)):
            feature = []

            for j in [i - 3, i - 2, i - 1, i, i + 1, i + 2, i + 3]:
                if j in range(len(sentence)):
                    feature.append(get_vec(sentence[i][0]))
                else:
                    feature.append(get_vec('space'))

            if sentence[i][3].endswith('O'):
                tag = np.asarray([1, 0, 0, 0, 0])
            elif sentence[i][3].endswith('PER'):
                tag = np.asarray([0, 1, 0, 0, 0])
            elif sentence[i][3].endswith('LOC'):
                tag = np.asarray([0, 0, 1, 0, 0])
            elif sentence[i][3].endswith('ORG'):
                tag = np.asarray([0, 0, 0, 1, 0])
            elif sentence[i][3].endswith('MISC'):
                tag = np.asarray([0, 0, 0, 0, 1])

            features.append(np.reshape(feature, (1, -1)))
            tags.append(tag)

    # 返回的时候把list转成(n_sample, s_feature)的格式 方便向网络传递参数
    return np.reshape(features, (len(features), -1)), np.reshape(tags, (len(tags), -1))


def next_batch(datasets, batch_size, sentence_size):
    current_block = 0
    current_batch = 0

    sentences = get_input()

    return 0


word_set = dict()
with open('myvectors.txt', 'r') as rFile:
    for line in rFile.readlines():
        arr = line.strip().split()
        word = arr[0]
        vec = arr[1:]
        word_set[word] = vec

print('word set length: ', len(word_set))

# train_x, train_y = get_input(data_path + '/eng.train')
# test_x, test_y = get_input(data_path + '/eng.testb'）

# print('train data size: ', len(train_x), len(train_y))
# print('test data size: ', len(test_x), len(test_y))

# graph input
x = tf.placeholder('float', [None, n_input])
y = tf.placeholder('float', [None, n_classes])


def mlp(x, weights, biases):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['h1'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['h2'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['h3'])
    layer_3 = tf.nn.sigmoid(layer_3)

    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    # out_layer = tf.nn.softmax(out_layer)
    return out_layer


def precision_score(y_pred, y_true):
    tp = [0] * 5
    p = [0] * 5
    for i in range(len(y_pred)):
        if y_true[i] == y_pred[i]:
            tp[y_pred[i]] += 1
        p[y_pred[i]] += 1
    # print('token numbers: ', p)
    print('total precision: ', np.sum(tp[1:] / np.sum(p[1:])))
    if p[1] > 0:
        return [tp[i] / p[i] for i in range(len(tp))]
    return 0


def recall_score(y_pred, y_true):
    tp = [0] * 5
    t = [0] * 5
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            tp[y_pred[i]] += 1
        t[y_true[i]] += 1
    # print('token numbers: ', t)
    print('total recall: ', np.sum(tp[1:] / np.sum(t[1:])))
    if t[1] > 0:
        return [tp[i] / t[i] for i in range(len(tp))]
    return 0


# weights
weights = {
    'h1': tf.Variable(tf.random_uniform([n_input, n_hidden_1], minval=- math.sqrt(6) / 40, maxval=math.sqrt(6) / 40)),
    'h2': tf.Variable(
        tf.random_uniform([n_hidden_1, n_hidden_2], minval=- math.sqrt(6) / 40, maxval=math.sqrt(6) / 40)),
    'h3': tf.Variable(
        tf.random_uniform([n_hidden_2, n_hidden_3], minval=- math.sqrt(6) / 40, maxval=math.sqrt(6) / 40)),
    'out': tf.Variable(tf.random_uniform([n_hidden_3, n_classes], minval=- math.sqrt(6) / 40, maxval=math.sqrt(6) / 40))
}

biases = {
    'h1': tf.Variable(tf.random_normal([n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

pred = mlp(x, weights, biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

if __name__ == "__main__":

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(training_epochs):

            for block_i in range(math.ceil(SENTENCE_SIZE / BLOCK_SIZE)):
                train_x, train_y = get_input(data_path + '/eng.train', block_i * BLOCK_SIZE,
                                             (block_i + 1) * BLOCK_SIZE)
                gc.collect()

                avg_cost = 0
                total_batch = math.ceil((len(train_x) / batch_size))

                batch_list = [index for index in range(total_batch)]
                random.shuffle(batch_list)
                for batch_i in batch_list:
                    _, c = sess.run([optimizer, cost],
                                    feed_dict={x: train_x[batch_i * batch_size:(batch_i + 1) * batch_size],
                                               y: train_y[batch_i * batch_size:(batch_i + 1) * batch_size]
                                               })
                    avg_cost += c / total_batch
                print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                      "{:.9f}".format(avg_cost))

            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print(accuracy.eval({x: test_x, y: test_y}))

            y_p = tf.argmax(pred, 1)

            val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x: train_x, y: train_y})
            y_true = np.argmax(train_y, 1)

            print('train precision: ', precision_score(y_pred, y_true))
            print('train recall: ', recall_score(y_pred, y_true))

            val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x: test_x, y: test_y})
            y_true = np.argmax(test_y, 1)

            print('test precision: ', precision_score(y_pred, y_true))
            print('test recall: ', recall_score(y_pred, y_true))
