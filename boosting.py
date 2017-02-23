import numpy as np
import math
from BaseEstimator import BaseEstimator, MultiLayerPerceptron
from crf import CRF
from conllner import read_data_set, read_test_data_set


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

        y_pred = estimator.predict(batch_x) # ont-hot (n_examples, n_classes)

        y_pred_scalar = np.argmax(y_pred, axis=1)

        y_true = crf.processor(batch_y)
        y_true_scalar = np.argmax(y_true, axis=1)
        weights = (1.0 / len(y_true)) * np.ones(len(y_true))

        return np.mean(weights * np.equal(y_pred_scalar, y_true_scalar))

    def _ensemble(self, estimators, trains):

        if len(estimators) != len(trains):
            raise ValueError('do not match')

        for i in range(len(estimators)):
            est = estimators[i]
            train = trains[i]

            est.predict(train)

        return


if __name__ == "__main__":
    train_crf = read_data_set('crf')
    train_w2v = read_data_set('w2v')
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
    crf.load()

    print(ada_boosting._err_rate(train_crf.tokens, train_crf.labels, crf, []))
    train_x, train_y = train_w2v.sent2features(train_w2v.tokens[:8080], train_w2v.labels[:8080])
    print(ada_boosting._err_rate(train_x, train_y, mlp, []))

