import matplotlib.pyplot as plt
import numpy as np

from optimization.kernel import Kernel


class GaussianProcessRgressor:

    def __init__(self, kernel="gaussian_kernel", sigma=1, l=1):
        self.kernel = kernel
        self.sigma = sigma
        self.l = l

        self.prior_mean = None
        self.prior_cov = None
        self.post_mean = None
        self.post_cov = None

        self.X = None
        self.y = None

    def fit(self, X, y):
        self.prior_mean, self.prior_cov = self.multivariate_norm_prior(X)
        self.y = y
        self.X = X

    def predict(self, X):
        if self.kernel == "gaussian_kernel":
            cov_XP_X = Kernel.gaussian_kernel(X, self.X)
            cov_X_X = Kernel.gaussian_kernel(X, X)
            cov_X_XP = cov_XP_X.T

        mean_x, cov_x = self.multivariate_norm_prior(X)
        K_Kinv = np.matmul(cov_XP_X, np.linalg.inv(self.prior_cov))
        self.post_mean = np.matmul(K_Kinv, self.y - self.prior_mean) + mean_x
        self.post_cov = cov_X_X - np.matmul(K_Kinv, cov_X_XP)

        return self.post_mean, self.post_cov

    def multivariate_norm_prior(self, xs):
        """
        Initialize the prior. The mean is, by default, set to zeros.
        """

        D = xs.shape[0]

        if self.kernel == "gaussian_kernel":
            cov = Kernel.gaussian_kernel(xs, sigma=self.sigma, l=self.l)
        else:
            cov = np.eye(D)

        mean = np.zeros(D)

        return mean, cov

    """
    ============================================    
    Following functions are used for test cases.
    ============================================
    """

    def sample_from_prior(self, mean, covariance):
        return np.random.multivariate_normal(mean, covariance)

    def plot_unit_gaussian_samples(self, D, n_sample):
        mean = np.array([i for i in range(D)])
        xs = np.linspace(0, 1, D).reshape((-1, 1))

        if self.kernel == "gaussian_kernel":
            cov = Kernel.gaussian_kernel(xs, sigma=self.sigma, l=self.l)
        else:
            cov = np.eye(D)

        for _ in range(n_sample):
            ys = self.sample_from_prior(mean, cov)
            plt.plot(xs, ys)

        plt.show()