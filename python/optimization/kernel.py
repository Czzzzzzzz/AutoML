import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform

class Kernel:

    """

    k(x1, x2) = sigma**2 * \exp{-(||x1 - x2||_2^2 / l)**2 / 2}

    Parameters
    ==========
    l: int
      l determines the degree of smoothness in great part. The smaller
      l is, the more smooth the curve is.

    Return
    ======
    cov: array-like, (N, N)
      covariance matrix
    mean: array-like, (N, )
      mean vector
    """
    @staticmethod
    def gaussian_kernel(x, y=None, sigma=1, l=1):
        N = x.shape[0]

        if y is None:
            dists = pdist(x, "euclidean")
            cov = np.exp(-0.5 * (dists / l) ** 2)
            cov = squareform(cov)
            np.fill_diagonal(cov, 1)
        else:
            dists = cdist(x , y, metric='euclidean')
            cov = np.exp(-.5 * (dists / l) ** 2)

        return cov

if __name__ == "__main__":

    x1 = np.array([[1, 2, 3], [4, 5, 6], [3, 3, 4]])
    Kernel.gaussian_kernel(x1)