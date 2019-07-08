import numpy as np

class Kernel:

    """
        k(x1, x2) = sigma**2 * \exp{-((x1 - x2) / l)**2 / 2}

        l: int
        l determines the degree of smoothness in great part. The smaller
        l is, the more smooth the curve is.
    """
    @staticmethod
    def gaussian_kernel(x1, x2, sigma=1, l=1):
        x1_mat = np.expand_dims(x1, 1)
        x2_mat = np.expand_dims(x2, 0)

        return sigma**2 * np.exp(-((x1_mat - x2_mat) / l)**2 / 2)

if __name__ == "__main__":

    print(Kernel.gaussian_kernel([1, 2], [1, 2, 3]))