from itertools import chain
from collections import Counter
import nltk
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
from load import get_sentences, precision_score, recall_score
from BaseEstimator import BaseEstimator
import conllner

from mlp import train

train_path = './data/eng.train'
train_sentences = get_sentences(train_path)
train_sentences = [[tuple(sentence[j]) for j in range(len(sentence))] for sentence in train_sentences]

test_path = './data/eng.testb'
test_sentences = get_sentences(test_path)
test_sentences = [[tuple(sentence[j]) for j in range(len(sentence))] for sentence in test_sentences]


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


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [label for _, _, _, label in sent]


def sent2tokens(sent):
    return [token for token, _, _, _ in sent]


def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.

    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))

    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}

    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels=[class_indices[cls] for cls in tagset],
        target_names=tagset,
        digits=4
    )


def processor(y_pred):
    # first to array
    y_final = []
    for i in range(len(y_pred)):
        y_final += y_pred[i]

    for i in range(len(y_final)):
        if y_final[i].endswith('O'):
            y_final[i] = np.asarray([1, 0, 0, 0, 0])
        elif y_final[i].endswith('PER'):
            y_final[i] = np.asarray([0, 1, 0, 0, 0])
        elif y_final[i].endswith('LOC'):
            y_final[i] = np.asarray([0, 0, 1, 0, 0])
        elif y_final[i].endswith('ORG'):
            y_final[i] = np.asarray([0, 0, 0, 1, 0])
        elif y_final[i].endswith('MISC'):
            y_final[i] = np.asarray([0, 0, 0, 0, 1])

    y_final = np.reshape(y_final, (len(y_final), 5))

    return y_final


def trans_prob(CRF_path, X, y):
    # crf_path : path for load trained crf path
    # X : list of sentence
    # y : list of labels

    # check input
    if len(X) != len(y):
        raise TypeError('invalid input value')

    # load crf model
    tagger = pycrfsuite.Tagger()
    tagger.open('./tmp/models/conll2003-eng.crfsuite')

    # some global parameters
    n_sentences = len(y)

    full_probability = []
    for i in range(n_sentences):
        tagger.set(X[i])
        for j in range(len(X[i])):
            # full_probability.append([tagger.marginal(tagger.labels()[k], j) for k in range(len(tagger.labels()))])
            full_probability.append(
                [tagger.marginal('O', j), tagger.marginal('I-PER', j),
                 tagger.marginal('I-LOC', j) + tagger.marginal('B-LOC', j),
                 tagger.marginal('I-ORG', j) + tagger.marginal('B-ORG', j),
                 tagger.marginal('I-MISC', j) + tagger.marginal('B-MISC', j)])

    # transform format
    full_probability = np.reshape(full_probability, (len(full_probability), -1))

    return full_probability


def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))


class CRF(BaseEstimator):
    def __init__(self,
                 c1=0.5,
                 c2=5e-04,
                 max_iteration=250,
                 model_dir = ''
                 ):
        self._nontrain = True
        self._trainer = pycrfsuite.Trainer(verbose=False)
        self._trainer.set_params(
            {
                'c1': 0.5,
                'c2': 5e-04,
                'max_iterations': 250,
                'feature.possible_transitions': True
            }
        )
        self._model_dir = model_dir

        return

    @staticmethod
    def processor(y_pred):
        # first to array
        y_final = []
        for i in range(len(y_pred)):
            y_final += y_pred[i]

        for i in range(len(y_final)):
            if y_final[i].endswith('O'):
                y_final[i] = np.asarray([1, 0, 0, 0, 0])
            elif y_final[i].endswith('PER'):
                y_final[i] = np.asarray([0, 1, 0, 0, 0])
            elif y_final[i].endswith('LOC'):
                y_final[i] = np.asarray([0, 0, 1, 0, 0])
            elif y_final[i].endswith('ORG'):
                y_final[i] = np.asarray([0, 0, 0, 1, 0])
            elif y_final[i].endswith('MISC'):
                y_final[i] = np.asarray([0, 0, 0, 0, 1])

        y_final = np.reshape(y_final, (len(y_final), 5))

        return y_final

    def fit(self, X, y):
        """train the model

        Arg:
            X : train data features
            y : train data labels
        """
        for train_x, train_y in zip(X, y):
            self._trainer.append(train_x, train_y)

        self._trainer.train(self._model_dir)
        self._nontrain = True

    def predict_sent(self, X):
        """ predict the sequence tag

        Arg:
            X : to be predicted sequence , list of sequence

        Return:
            y_pred :
        """
        tagger = pycrfsuite.Tagger()
        tagger.open(self._model_dir)

        y_pred = [tagger.tag(xseq) for xseq in X]
        return y_pred

    def predict(self, X):
        """

        :param X:
        :return:
        """
        y_pred_sent = self.predict_sent(X)



        return processor(y_pred_sent)

    def evaluate(self, X, y):
        """ measure of the model including precision, recall and f1-score

        Arg:
            X : feature
            y : label

        """

        y_pred = self.predict_sent(X)
        print(bio_classification_report(y, y_pred))

        return

if __name__ == "__main__":
    crf = CRF(model_dir='./tmp/models/conll2003-eng.crfsuite')
    conll_2003_train = conllner.read_data_set('crf')
    conll_2003_test = conllner.read_test_data_set('crf')
    crf.fit(conll_2003_train.tokens, conll_2003_train.labels)

    crf.evaluate(conll_2003_test.tokens, conll_2003_test.labels)
    y_pred = crf.predict(conll_2003_test.tokens)
    print("end")
    # train_x = [sent2features(s) for s in train_sentences]
    # train_y = [sent2labels(s) for s in train_sentences]
    #
    # test_x = [sent2features(s) for s in test_sentences]
    # test_y = [sent2labels(s) for s in test_sentences]
    #
    # trainer = pycrfsuite.Trainer(verbose=False)
    #
    # trainer.set_params(
    #     {
    #         'c1': 0.5,
    #         'c2': 5e-04,
    #         'max_iterations': 250,
    #         'feature.possible_transitions': True
    #     }
    # )
    #
    # for x, y in zip(train_x, train_y):
    #     trainer.append(x, y)
    #
    # trainer.train('./tmp/models/conll2003-eng.crfsuite')
    # print(trainer.logparser.last_iteration)
    #
    # tagger = pycrfsuite.Tagger()
    # tagger.open('./tmp/models/conll2003-eng.crfsuite')
    # crf_path = './tmp/models/conll2003-eng.crfsuite'
    #
    # print(tagger.labels())
    #
    # y_pred_mid = trans_prob(crf_path, test_x, test_y)
    # # y_pred_mid = np.argmax(y_pred_mid, 1)
    #
    # # for i in range(len(y_pred_mid)):
    # #     if y_pred_mid[i] == 1 or y_pred_mid[i] == 7:
    # #         y_pred_mid[i] = 3
    # #     elif y_pred_mid[i] == 2 or y_pred_mid[i] == 6:
    # #         y_pred_mid[i] = 4
    # #     elif y_pred_mid[i] == 3:
    # #         y_pred_mid[i] = 1
    # #     elif y_pred_mid[i] == 4 or y_pred_mid[i] == 5:
    # #         y_pred_mid[i] = 2
    #
    # # example_sent = test_sentences[12]
    # #
    # # print(' '.join(sent2tokens(example_sent)), end='\n\n')
    # # print("Predicted:", ' '.join(tagger.tag(sent2features(example_sent))))
    # # print("Correct:  ", ' '.join(sent2labels(example_sent)))
    #
    # y_pred = [tagger.tag(xseq) for xseq in test_x]
    # print(bio_classification_report(test_y, y_pred))
    #
    # # y_pred = [tagger.tag(test_x[i]) for i in range(len(test_x))]
    # # y_pred = processor(y_pred)
    # # y_pred = np.argmax(y_pred, 1)
    #
    # # y_p_mlp, y_true = train(flag=1)
    #
    # # y_all = np.argmax(y_p_mlp + y_pred_mid, 1)
    # #
    # # # print(bio_classification_report(test_y, y_pred))
    # #
    # #
    # # print('all; ', precision_score(y_all, y_true))
    # # print('all:' , recall_score(y_all, y_true))


