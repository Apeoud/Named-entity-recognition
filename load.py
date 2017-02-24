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
                    feature.append(get_vec(sentence[j][0]))
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

            # if sentence[i][3] == 'O':
            #     tag = np.asarray([1, 0, 0, 0, 0, 0, 0, 0])
            # elif sentence[i][3] == 'I-PER':
            #     tag = np.asarray([0, 1, 0, 0, 0, 0, 0, 0])
            # elif sentence[i][3] == 'B-LOC':
            #     tag = np.asarray([0, 0, 1, 0, 0, 0, 0, 0])
            # elif sentence[i][3] == 'I-LOC':
            #     tag = np.asarray([0, 0, 0, 1, 0, 0, 0, 0])
            # elif sentence[i][3] == 'B-ORG':
            #     tag = np.asarray([0, 0, 0, 0, 1, 0, 0, 0])
            # elif sentence[i][3] == 'I-ORG':
            #     tag = np.asarray([0, 0, 0, 0, 0, 1, 0, 0])
            # elif sentence[i][3] == 'B-MISC':
            #     tag = np.asarray([0, 0, 0, 0, 0, 0, 1, 0])
            # elif sentence[i][3] == 'I-MISC':
            #     tag = np.asarray([0, 0, 0, 0, 0, 0, 0, 1])

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
with open('./tmp/myvectors.txt', 'r') as rFile:
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
    print('token numbers: ', p)
    print('total precision: ', np.sum(tp[1:] / np.sum(p[1:])))
    return np.sum(tp[1:] / np.sum(p[1:]))
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
    print('token numbers: ', t)
    print('total recall: ', np.sum(tp[1:] / np.sum(t[1:])))
    return np.sum(tp[1:] / np.sum(t[1:]))
    if t[1] > 0:
        return [tp[i] / t[i] for i in range(len(tp))]
    return 0


if __name__ == "__main__":
    sentences = get_sentences(data_path + '/eng.train')
    tokens = []
    for i in range(len(sentences)):
        tokens += (sentences[i])
    print(len(tokens))
    matrix = np.reshape(tokens, (-1, 4))
    tags = matrix[:, -1]
    print(set(tags))
