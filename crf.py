from itertools import chain
from collections import Counter
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import pycrfsuite
from load import get_sentences

train_path = './data/eng.train'
train_sentences = get_sentences(train_path)
train_sentences = [[tuple(sentence[j]) for j in range(len(sentence))] for sentence in train_sentences]

test_path = './data/eng.train'
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
    )



def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

if __name__ == "__main__":
    train_x = [sent2features(s) for s in train_sentences]
    train_y = [sent2labels(s) for s in train_sentences]

    test_x = [sent2features(s) for s in test_sentences]
    test_y = [sent2labels(s) for s in test_sentences]

    trainer = pycrfsuite.Trainer(verbose=False)

    for x, y in zip(train_x, train_y):
        trainer.append(x, y)

    trainer.set_params(
        {
            'c1' : 1.0,
            'c2' : 1e-03,
            'max_iterations' : 50,
            'feature.possible_transitions' : True
        }
    )

    # trainer.train('conll2003-eng.crfsuite')
    # print(trainer.logparser.last_iteration)

    tagger = pycrfsuite.Tagger()
    tagger.open('conll2003-eng.crfsuite')
    info = tagger.info()
    print("Top likely transitions:")
    print_transitions(Counter(info.transitions).most_common(15))

    example_sent = test_sentences[12]

    print(' '.join(sent2tokens(example_sent)), end='\n\n')
    print("Predicted:", ' '.join(tagger.tag(sent2features(example_sent))))
    print("Correct:  ", ' '.join(sent2labels(example_sent)))


    y_pred = [tagger.tag(xseq) for xseq in test_x]
    print(bio_classification_report(test_y, y_pred))


