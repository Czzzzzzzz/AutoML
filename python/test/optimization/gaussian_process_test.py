import unittest
import numpy as np
import matplotlib.pyplot as plt
from optimization.gaussian_process import *



class GaussianProcessTest(unittest.TestCase):

    def test_sample_from_multinorm(self):
        gp = GaussianProcessRgressor(l=1)
        gp.plot_unit_gaussian_samples(10, 5)

    def test_gaussian_regression(self):
        xs = np.array([-4, -1.5, 0, 1.5, 2.5, 2.7]).reshape((-1, 1))
        ys = np.array([self.target_function(x) for x in xs])
        xs_pred = np.linspace(-5, 3.5, 80).reshape((-1, 1))
        ys_pred = np.array([self.target_function(x) for x in xs_pred])

        gp = GaussianProcessRgressor(l=1)
        gp.fit(xs, ys)
        mean, cov = gp.predict(xs_pred)
        sigma = np.sqrt(np.diag(cov))

        xs = np.array([-4, -1.5, 0, 1.5, 2.5, 2.7])
        xs_pred = np.linspace(-5, 3.5, 80)

        plt.plot(xs, ys, marker='o', color='blue', label='traning')
        plt.plot(xs_pred, ys_pred, color='orange', label='true')
        plt.plot(xs_pred, mean, color="red", label='prediction')
        plt.fill_between(xs_pred, mean - 2*sigma, mean + 2*sigma, color='grey', label='uncertainty')
        plt.legend()
        plt.show()

    def target_function(self, xs):
        y = 0
        for idx, x in enumerate(xs):
            y += x**idx + 2 * x ** (idx+1)
        return y
