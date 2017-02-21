import numpy as np
import re

def extract_sentences_labels(path):
    """extract sentences into a list of sentence

    Args:
        path : a file path that has the following format
                UU          NNP I-NP    I-ORG
                official    NN  I-NP    O
                Ekeus       NNP I-NP    I-PER

    Returns:
        sentences : list of list of tokens e.g. [['today', 'Bob', 'has', ], ... , []]
        labels : list of list of labels    e.g. [['O', 'I-PER', 'O'], ... , []]
    """

    print('Extracting ', path)

    sentences = []
    sentence = []
    sentences_labels = []
    sentence_labels = []

    with open(path, 'r') as rFile:
        for line in rFile.readlines():
            arr = line.strip().split()
            if len(arr) != 4:
                sentences.append(sentence)
                sentence = []

                sentences_labels.append(sentence_labels)
                sentence_labels = []
            else:
                sentence.append(arr[0])
                sentence_labels.append(arr[3])

    # return list of sentences [sent0, sent1, ... , sent100]
    # sent0 = ['today', 'i', 'want', ...]

    return sentences, sentences_labels


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


class DataSet(object):
    def __init__(self,
                 tokens,
                 labels,
                 one_hot=False):
        # assert tokens.shape[0] == labels.shape[0], ('tokens.shape: %s labels.shape: %s' % (tokens.shape, labels.shape))
        self._num_examples = len(tokens)

        self._tokens = tokens
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch

        # start epoch with shuffle
        if self._epochs_completed == 0 and start == 0 and shuffle:
            # permute the data
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._tokens = [self._tokens[i] for i in range(len(perm0))]
            self._labels = [self._labels[i] for i in range(len(perm0))]

        # go to the next batch
        if start + batch_size > self._num_examples:
            rest_num_examples = self._num_examples - start

            token_rest_part = self._tokens[start:self._num_examples]
            label_rest_part = self._labels[start:self._num_examples]

            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._tokens = [self._tokens[i] for i in range(len(perm))]
                self._labels = [self._labels[i] for i in range(len(perm))]

            start = 0

            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            token_new_part = self._tokens[start:end]
            label_new_part = self._labels[start:end]

            return np.concatenate((token_rest_part, token_new_part)), np.concatenate((label_rest_part, label_new_part))
        else:
            # return the next batch
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._tokens[start:end], self._labels[start:end]

    def sent2features(self, batch_x, batch_y):

        return


class Word_Window_DataSet(DataSet):
    def __init__(self,
                 tokens,
                 labels,
                 dictionary_path,
                 window_size = 5,
                 vector_size = 300,
                 one_hot=False):
        super().__init__(tokens, labels, one_hot)

        self._windows_size = window_size
        self._vector_size = vector_size
        # load the dictionary from saved word2vec model
        self.dictionary = dict()
        with open(dictionary_path, 'r') as rFile:
            for line in rFile.readlines():
                arr = line.strip().split()
                word = arr[0]
                vec = arr[1:]
                self.dictionary[word] = vec


    def word2vec(self, word):
        rand = np.random.uniform(-0.25, 0.25, self._vector_size)
        s = re.sub('[^0-9a-zA-Z]+', '', word)
        vec = []
        if word == 'space':
            vec = [0 for _ in range(self._vector_size)]
        elif word in self.dictionary:
            vec = self.dictionary[word]
        elif word.lower() in self.dictionary:
            vec = self.dictionary[word.lower()]
        elif s in self.dictionary:
            vec = self.dictionary[s]
        elif s.isdigit():
            vec = self.dictionary['1']

        if len(vec) > 0:
            return np.asarray(vec)
        return rand


    def sent2features(self, batch_x, batch_y):
        """ according different models using different features selection methods
            in word window method
        :param batch_x:
        :param batch_y:
        :return:
        """

        features = []
        tags = []
        # sentences = get_sentences(filename)[sent_begin:sent_end]
        for sentence, label in zip(batch_x, batch_y):
            for i in range(len(sentence)):
                feature = []

                for j in [i - k + (self._windows_size - 1) /2 for k in range(self._windows_size)]:
                    if j in range(len(sentence)):
                        feature.append(self.word2vec(sentence[i][0]))
                    else:
                        feature.append(self.word2vec('space'))

                if label[i].endswith('O'):
                    tag = np.asarray([1, 0, 0, 0, 0])
                elif label[i].endswith('PER'):
                    tag = np.asarray([0, 1, 0, 0, 0])
                elif label[i].endswith('LOC'):
                    tag = np.asarray([0, 0, 1, 0, 0])
                elif label[i].endswith('ORG'):
                    tag = np.asarray([0, 0, 0, 1, 0])
                elif label[i].endswith('MISC'):
                    tag = np.asarray([0, 0, 0, 0, 1])


                features.append(np.reshape(feature, (1, -1)))
                tags.append(tag)

        return np.reshape(features, (len(features), -1)), np.reshape(tags, (len(tags), -1))

TRAIN_PATH = './data/eng.train'
VALIDATION_PATH = './data/eng.testa'
TEST_PATH = './data/eng.testb'
DICT_PATH = './tmp/myvectors.txt'

sentences, labels = extract_sentences_labels(TRAIN_PATH)

dataset = DataSet(tokens=sentences, labels=labels)
window_nn = Word_Window_DataSet(tokens=sentences, labels=labels, dictionary_path=DICT_PATH)

batch_x, batch_y = window_nn.next_batch(10,True)
print(batch_x)
print(batch_y)

batch_x, batch_y = window_nn.next_batch(10,True)
print(batch_x)
print(batch_y)

train_x, train_y = window_nn.sent2features(batch_x, batch_y)

print('end')