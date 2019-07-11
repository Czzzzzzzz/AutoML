from optimization.gaussian_process import GaussianProcessRgressor
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_validate, cross_val_score
from scipy.stats import norm

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
        self.NAME = "name"
        self.TYPE_CONTINUOUS = "continuous"
        self.TYPE_CATEGORICAL = "categorical"
        self.DOMAIN = "domain"

        self.domains = domains
        self.f = f
        self.max_iterations = max_iterations
        self.base_estimator = base_estimator
        self.acquisition = acquisition

        # Establish the mapping table from parameter names to indices
        self.__build_mapping_table()

    def __build_mapping_table(self):
        self.name2idx = {}
        self.idx2name = {}
        for idx, param in enumerate(self.domains):
            self.name2idx[param[self.NAME]] = idx
            self.idx2name[idx] = param[self.NAME]

    def __paramVec2ParamDic(self, param_vec):
        param_dic = {}
        for idx, value in enumerate(param_vec):
            param_dic[self.idx2name[idx]] = value
        return param_dic

    def init_samples(self, data, label, num):
        """
        Return
        ------
        samples: dictionary
            key is the name of parameter
            value is the value of the corresponding value of the parameter
        y: np.array, 1 dimensions
        """
        samples = np.zeros((num, len(self.domains)))
        for idx, param in enumerate(self.domains):
            feature = np.random.uniform(param[self.DOMAIN][0], param[self.DOMAIN][1], num)
            samples[:, self.name2idx[param[self.NAME]]] = feature

        ys = np.array([self.f(data, label, self.__paramVec2ParamDic(samples[idx])) for idx in range(num)])
        return samples, ys

    def expected_improvement(self, Xs, ys, optimal_y, mean, cov):
        EIs = []
        for idx, prediction_y in enumerate(mean):
            prediction_dev = np.sqrt(cov[idx, idx])
            normalization = (optimal_y - prediction_y) / prediction_dev
            ei = (optimal_y - prediction_y) * norm.cdf(normalization) + prediction_dev * norm.pdf(normalization)
            EIs.append(ei)
        optimal_idx = np.argmax(EIs)
        return Xs[optimal_idx], ys[optimal_idx]

    def propose_next_sample(self, data, label, optimal_y, gp):
        new_samples, ys = self.init_samples(data, label, 10)
        if self.acquisition == "EI":
            mean, cov = gp.predict(new_samples)
            optimal_sample, y = self.expected_improvement(new_samples, ys, optimal_y, mean, cov)
        return optimal_sample, y

    def run(self, data, label):
        """
        Init samples
        for i in range(iterations):
            Update posterior distribution by gaussian process.
            Find the next sample maximizing acquisition function.
            Add new sample to the original samples.
        The last new sample is the best result.
        """

        # initial parameters
        samples, ys = self.init_samples(data, label, 10)
        optimal_y = np.min(ys)

        # define the gp
        gp = self.base_estimator
        for _ in range(self.max_iterations):

            # Fit gp on the hyperparameters.
            gp.fit(samples, ys)

            # Sample next hyperparameter.
            next_sample, y = self.propose_next_sample(data, label, optimal_y, gp)

            # Update samples
            samples = np.r_([samples, next_sample])
            ys = np.r_([ys, y])
            optimal_y = np.min(ys)

        return samples, ys


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
        func_value += coef * x**power
        Xs.append(x)

    return np.array(Xs), func_value


if __name__ == "__main__":

    np.random.seed(0)
    Xs, ys = generate_data_from_func(target_function, 1000, 5)

    parameters = np.array([{"name": "alpha", "type": "continuous", "domain": [0, 10]},
                           {"name": "l1_ratio", "type": "continuous", "domain": [0, 10]}])
    bayes_opti = BayesianOptimization(score, parameters, max_iterations=1)
    bayes_opti.run(Xs, ys)

