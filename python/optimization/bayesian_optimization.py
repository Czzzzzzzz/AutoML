from optimization.gaussian_process import GaussianProcessRgressor
from sklearn.gaussian_process import GaussianProcessClassifier

from utils.data_generator import generate_data_from_func

import numpy as np

class BayesianOptimization:

    def __init__(self, f, domains, max_iterations=20, base_estimator=GaussianProcessRgressor(), acquisition = 'EI'):
        """
        Parameters
        ----------
        f: function
            return the score of black box function over the input.

        domains: np.array[dictionary]
            The format of domains should be like {"name": xxx, "type": xxx, "domain": xxx}
            i.e. bds = [{'name': 'learning_rate', 'type': 'continuous', 'domain': (0, 1)},
                        {'name': 'gamma', 'type': 'continuous', 'domain': (0, 5)}]

        base_estimator: object, GaussianProcessRegressor

        acqusition: string, 'EI'
        """
        self.domains = domains
        self.f = f
        self.max_iterations = max_iterations
        self.base_estimator = base_estimator
        self.acquisition = acquisition

    def init_samples(self, num):
        """

        Return
        ------
        samples: np.array, 2 dimensions
        y: np.array, 1 dimensions
        """
        samples = None
        ys = []
        for idx, param in enumerate(self.domains):
            feature = np.random.uniform(param['domain'][0], param['domain'][1], num)
            if samples is None:
                samples = feature
            else:
                samples = np.vstack((samples, feature))

            ys.append(self.score(feature))

        samples = samples.T
        ys = np.array(ys)
        return samples, ys

    def expected_improvement(self, Xs, ys, mean, cov):
        return Xs[0], ys[0]

    def propose_next_sample(self, gp):
        new_samples, ys = self.init_samples(10)
        if self.acquisition == "EI":
            mean, cov = gp.predict(new_samples)
            optimal_sample, y = self.expected_improvement(new_samples, ys, mean, cov)

        return optimal_sample, y

    def run(self, X, y):
        """
        Init samples
        for i in range(iterations):
            Update posterior distribution by gaussian process.
            Find the next sample maximizing acquisition function.
            Add new sample to the original samples.
        The last new sample is the best result.
        """

        # initial parameters
        samples, ys = self.init_samples(10)

        # define the gp
        gp = self.base_estimator
        for _ in range(self.max_iterations):

            # Fit gp on the hyperparameters.
            gp.fit(samples, ys)

            # Sample next hyperparameter.
            next_sample, y = self.propose_next_sample(gp)

            # Update samples
            samples = np.r_([samples, next_sample])
            ys = np.r_([ys, y])

        return samples, ys


def score(x):
    return 1


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
        func_value += coef * x**power
        Xs.append(x)

    return np.array(Xs), func_value


if __name__ == "__main__":

    np.random.seed(0)
    Xs, ys = generate_data_from_func(target_function, 20, 2)

    parameters = np.array([{"name": "learning_rate", "type": "continuous", "domain": [-1, 1]},
                           {"name": "alpha", "type": "continuous", "domain": [1, 10]}])
    bayes_opti = BayesianOptimization(score, parameters)
    bayes_opti.run(Xs, ys)


