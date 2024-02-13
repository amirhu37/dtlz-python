import numpy as np
from dtlz_dimension_check import dtlz_dimension_check



def dtlz1(x, M):
    # Check input dimension
    k = 5  # As suggested by Deb
    dtlz_dimension_check(x, M, k)
    n = (M - 1) + k
    xm = x[n - k :]

    g = 100 * (k + np.sum((xm - 0.5) ** 2 - np.cos(20 * np.pi * (xm - 0.5))))

    # Compute the functions
    fx = np.zeros((M, x.shape[1]))

    # The first and the last will be written separately to facilitate things
    fx[0, :] = 0.5 * np.prod(x[:M - 1, :], axis=0) * (1 + g)
    for ii in range(1, M - 1):
        fx[ii, :] = 0.5 * np.prod(x[:M - ii, :], axis=0) * (1 - x[M - ii, :]) * (1 + g)
    fx[M - 1, :] = 0.5 * (1 - x[0, :]) * (1 + g)

    return fx


import matplotlib.pyplot as plt
import seaborn as sns
if __name__ == "__main__":
    N = 100
    x1 = np.linspace(0,1, N)
    x2to5 = np.tile(.5, [5, N])
    x = np.vstack((x1, x2to5))
    fx = dtlz1(x, 2)
    sns.lineplot(fx, legend=False)
    plt.show()