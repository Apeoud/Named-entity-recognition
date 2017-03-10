import numpy as np
import math
from BaseEstimator import BaseEstimator, MultiLayerPerceptron
from crf import CRF
from conllner import read_data_set, read_test_data_set
from load import precision_score, recall_score


class AdaBoosting(object):
    def __init__(self):
        return


    def fit(self, train_x, train_y):
        # checking input
        # global parameters
        n_samples, n_features = train_x.shape
        n_classes = train_y[1]

        # initializing weights time update
        weights = (1.0 / n_samples) / np.ones(n_samples)


    def _indicator_func(self, a, b):
        """ indicator function """
        if a == b:
            return True
        else:
            return False

    def _err_rate(self, batch_x, batch_y, estimator, weights):
        """ calculate error rate using different weights"""

        # if len(batch_x) != len(weights):
        #     raise ValueError("invalid shape")

        if not isinstance(estimator, BaseEstimator):
            raise ValueError("not estimator")

        y_pred = estimator.predict(batch_x) # ont-hot (n_examples, )

        # y_pred_scalar = np.argmax(y_pred, axis=1)


        y_true_scalar = np.argmax(batch_y, axis=1)
        weights = (1.0 / len(y_true_scalar)) * np.ones(len(y_true_scalar))

        return np.mean(weights * np.equal(y_pred, y_true_scalar))

    def _ensemble(self, estimators, X, y, weights):

        if len(estimators) != len(X):
            raise ValueError('do not match')

        pred = np.zeros((len(X[1]), 5))

        for i in range(len(estimators)):
            est = estimators[i]
            train = X[i]

            y_pred = est.proba(train)
            pred += weights[i] * y_pred

            # print(precision_score(y_pred, y[i]))

        precision = precision_score(np.argmax(pred, 1), np.argmax(y[0], 1))
        recall = recall_score(np.argmax(pred, 1), np.argmax(y[0], 1))
        print('recall : ', 2 * precision * recall / (precision + recall))


        return


if __name__ == "__main__":
    train_crf = read_test_data_set('crf')
    train_w2v = read_test_data_set('w2v')
    ada_boosting = AdaBoosting()

    crf = CRF(model_dir='./tmp/models/conll2003-eng.crfsuite')
    mlp = MultiLayerPerceptron(
        hidden_units=[300, 300],
        train_epochs=5,
        n_input=2100,
        n_classes=5,
        batch_size=100,
        learning_rate=0.005,
        model_dir='./tmp/models/tf_mlp.ckpt'
    )

    est = []
    est_x = []
    est_y = []


    train_x = train_crf.sent2features(train_crf.data)
    train_y = train_crf.sent2label(train_crf.data)
    train_y = crf.processor(train_y)

    est.append(crf)
    est_x.append(train_x)
    est_y.append(train_y)

    # print(ada_boosting._err_rate(train_x, train_y, crf, []))
    train_x = train_w2v._extract_feature(train_w2v.data)
    train_y = train_w2v._extract_label(train_w2v.data)

    est.append(mlp)
    est_x.append(train_x)
    est_y.append(train_y)

    # print(ada_boosting._err_rate(train_x, train_y, mlp, []))
    ada_boosting._ensemble(est, est_x, est_y, weights=[0.70, 0.3])
