import unittest

import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score
from optimization.bayesian_optimization import BayesianOptimization

from utils.data_generator import generate_data_from_func


def score(X, y, parameters):
    clf = ElasticNet(**parameters)
    score_ = cross_val_score(clf, X, y, cv=5, scoring="neg_mean_squared_error")
    return score_


def target_function(feature_size):
    """
    The target function which is used to generate data.
    """

    func_value = 0
    Xs = []
    for i in range(feature_size):
        power = i % 3 + 1
        coef = np.random.randint(1, 10)
        x = np.random.randint(1, 5)
        func_value += coef * x ** power
        Xs.append(x)

    return np.array(Xs), func_value

class BayesianOptimizationTest(unittest.TestCase):

    def test_bayesian_optimization(self):
        np.random.seed(0)
        Xs, ys = generate_data_from_func(target_function, 1000, 5)

        parameters = np.array([{"name": "alpha", "type": "continuous", "domain": [0, 10]},
                               {"name": "l1_ratio", "type": "continuous", "domain": [0, 10]}])
        bayes_opti = BayesianOptimization(score, parameters, max_iterations=1)
        bayes_opti.run(Xs, ys)
