import numpy as np
from dtlz_dimension_check import dtlz_dimension_check

def dtlz2(x, M):
    # Check input dimension
    k = 10
    dtlz_dimension_check(x, M, k)
    n = (M - 1) + k
    xm = x[n - k :]

    g = np.sum((xm - 0.5) ** 2, axis=0)

    # Compute the functions
    fx = np.zeros((M, x.shape[1]))

    fx[0, :] = (1 + g) * np.prod(np.cos(np.pi / 2 * x[:M - 1, :]), axis=0)
    for ii in range(1, M - 1):
        fx[ii, :] = (1 + g) * np.prod(np.cos(np.pi / 2 * x[:M - ii, :]), axis=0) * np.sin(np.pi / 2 * x[M - ii, :])
    fx[M - 1, :] = (1 + g) * np.sin(np.pi / 2 * x[0, :])

    return fx


import matplotlib.pyplot as plt
import seaborn as sns
if __name__ == "__main__":
    N = 100
    x1 = np.linspace(0,1, N)
    x2to11 = np.tile(.5, [10, N])
    x = np.vstack((x1, x2to11))
    fx = dtlz2(x, 2)
    sns.lineplot(fx, legend=False)
    plt.show()