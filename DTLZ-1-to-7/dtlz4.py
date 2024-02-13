from dtlz_dimension_check import dtlz_dimension_check
import numpy as np

def dtlz4(x, M):
    k = 10
    alpha = 100
    dtlz_dimension_check(x, M, k)

    xm = x[-k:, :]  # xm contains the last k variables
    g = np.sum((xm - 0.5) ** 2, axis=0)

    # Compute the functions
    fx = np.empty((M, x.shape[1]))
    fx[0, :] = (1 + g) * np.prod(np.cos(np.pi / 2 * x[:M - 1, :] ** alpha), axis=0)
    for ii in range(1, M - 1):
        fx[ii, :] = (1 + g) * np.prod(np.cos(np.pi / 2 * x[:M - ii, :] ** alpha), axis=0) * \
                    np.sin(np.pi / 2 * x[M - ii, :] ** alpha)
    fx[-1, :] = (1 + g) * np.sin(np.pi / 2 * x[0, :] ** alpha)

    return fx


import matplotlib.pyplot as plt
import seaborn as sns
if __name__ == "__main__":
    N = 100
    x1 = np.linspace(0,1, N)
    x2to11 = np.tile(.5, [10, N])
    x = np.vstack((x1, x2to11))
    fx = dtlz4(x, 2)
    sns.lineplot(fx, legend=False)
    plt.show()