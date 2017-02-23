from sklearn.svm import NuSVC
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier

from load import get_sentences, get_input

train_path = './data/eng.train'
train_sentences = get_sentences(train_path)
train_x, train_y = get_input(train_sentences[:1000])

test_path = './data/eng.testb'
test_sentences = get_sentences(test_path)
test_x, test_y = get_input(test_sentences[:400])

train_y = np.argmax(train_y, 1)
print(train_y)

print(train_x.shape, train_y.shape)

clf = AdaBoostClassifier(n_estimators=200)
clf.fit(train_x, train_y)

print(cross_val_score(clf, test_x, np.argmax(test_y, 1)))
