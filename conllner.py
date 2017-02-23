import numpy as np
import re


def extract_sentences_labels(path, feature=False):
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
                if feature:
                    sentence.append(arr)
                    sentence_labels.append(arr[3])
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

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def tokens(self):
        return self._tokens

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch

        # start epoch with shuffle
        if self._epochs_completed == 0 and start == 0 and shuffle:
            # permute the data
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._tokens = [self._tokens[perm0[i]] for i in range(len(perm0))]
            self._labels = [self._labels[perm0[i]] for i in range(len(perm0))]

        # go to the next batch
        if start + batch_size > self._num_examples:
            self._epochs_completed += 1
            rest_num_examples = self._num_examples - start

            token_rest_part = self._tokens[start:self._num_examples]
            label_rest_part = self._labels[start:self._num_examples]

            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._tokens = [self._tokens[perm[i]] for i in range(len(perm))]
                self._labels = [self._labels[perm[i]] for i in range(len(perm))]

            start = 0

            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            token_new_part = self._tokens[start:end]
            label_new_part = self._labels[start:end]

            # print("satrt, end : ", start, end)
            return np.concatenate((token_rest_part, token_new_part)), np.concatenate((label_rest_part, label_new_part))
        else:
            # return the next batch
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            # print("satrt, end : ", start, end)
            return self._tokens[start:end], self._labels[start:end]

    def sent2features(self, batch_x, batch_y):

        return


class Word_Window_DataSet(DataSet):
    def __init__(self,
                 tokens,
                 labels,
                 dictionary_path,
                 window_size=5,
                 vector_size=300,
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

                for j in [i + k - (self._windows_size - 1) / 2 for k in range(self._windows_size)]:
                    if j in range(len(sentence)):
                        feature.append(self.word2vec(sentence[int(j)]))
                    else:
                        feature.append(self.word2vec('space'))

                try:
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
                except Exception as e:
                    print(e)

                features.append(np.reshape(feature, (1, -1)))
                tags.append(tag)

        return np.reshape(features, (len(features), -1)), np.reshape(tags, (len(tags), -1))

    def next_batch(self, batch_size, shuffle=True):
        batch_x, batch_y = super().next_batch(batch_size, shuffle = False)
        batch_x, batch_y = self.sent2features(batch_x, batch_y)

        return batch_x, batch_y


class CRFdataset(DataSet):
    def __init__(self,
                 tokens,
                 labels,
                 ):
        self._tokens = self.sent2features(tokens)
        self._labels = labels

        return

    @staticmethod
    def word2features(sent, i):
        word = sent[i][0]
        pos_tag = sent[i][1]
        chunk_tag = sent[i][2]

        features = [
            'bias',
            'word.lower=' + word.lower(),
            'word[-3:]=' + word[-3:],
            'word[-2:]=' + word[-2:],
            'word.isupper=%s' % word.isupper(),
            'word.istitle=%s' % word.istitle(),
            'word.isdigit=%s' % word.isdigit(),
            'postag=' + pos_tag,
            'chunktag=' + chunk_tag,
            'postag[:2]=' + pos_tag[:2],
        ]

        if i > 0:
            word1 = sent[i - 1][0]
            postag1 = sent[i - 1][1]
            chunktag1 = sent[i - 1][2]
            features.extend([
                '-1:word.lower=' + word1.lower(),
                '-1:word.istitle=%s' % word1.istitle(),
                '-1:word.isupper=%s' % word1.isupper(),
                '-1:postag=' + postag1,
                '-1:chunktag=' + chunktag1,
                '-1:postag[:2]=' + postag1[:2],
            ])
        else:
            features.append('BOS')

        if i < len(sent) - 1:
            word1 = sent[i + 1][0]
            postag1 = sent[i + 1][1]
            chunktag1 = sent[i + 1][2]
            features.extend([
                '+1:word.lower=' + word1.lower(),
                '+1:word.istitle=%s' % word1.istitle(),
                '+1:word.isupper=%s' % word1.isupper(),
                '+1:postag=' + postag1,
                '+1:chunktag=' + chunktag1,
                '+1:postag[:2]=' + postag1[:2],
            ])
        else:
            features.append('EOS')

        return features

    def sent2features(self, batch_x):

        return [[self.word2features(batch_x[j], i) for i in range(len(batch_x[j]))] for j in range(len(batch_x))]

    def sent2label(self, batch_y):
        return [[label for _, _, _, label in batch_y[i]] for i in range(len(batch_y))]


def read_data_set(model_name):
    TRAIN_PATH = './data/eng.train'
    VALIDATION_PATH = './data/eng.testa'
    TEST_PATH = './data/eng.testb'
    DICT_PATH = './tmp/myvectors.txt'

    if model_name == 'crf':
        sentences_fea, labels = extract_sentences_labels(TRAIN_PATH, feature=True)
        return CRFdataset(tokens=sentences_fea, labels=labels)
    elif model_name == 'w2v':
        sentences, labels = extract_sentences_labels(TRAIN_PATH)
        return Word_Window_DataSet(tokens=sentences, labels=labels, window_size=7, dictionary_path=DICT_PATH)
    else:
        sentences, labels = extract_sentences_labels(TRAIN_PATH)
        return DataSet(tokens=sentences, labels=labels)


def read_test_data_set(model_name):
    TEST_PATH = './data/eng.testb'
    DICT_PATH = './tmp/myvectors.txt'
    if model_name == 'crf':
        sentences_fea, labels = extract_sentences_labels(TEST_PATH, feature=True)
        return CRFdataset(tokens=sentences_fea, labels=labels)
    elif model_name == 'w2v':
        sentences, labels = extract_sentences_labels(TEST_PATH)
        return Word_Window_DataSet(tokens=sentences, labels=labels, window_size=7, dictionary_path=DICT_PATH)
    else:
        sentences, labels = extract_sentences_labels(TEST_PATH)
        return DataSet(tokens=sentences, labels=labels)


if __name__ == "__main__":
    conll = read_data_set('w2v')
    while True:
        batch_x, batch_y = conll.next_batch(500)
        conll.sent2features(batch_x, batch_y)
        print(batch_x.shape, batch_y.shape)
