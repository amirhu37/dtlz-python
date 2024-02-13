from dtlz_dimension_check import dtlz_dimension_check
import numpy as np

def dtlz7(x, M):
    k = 20

    dtlz_dimension_check(x, M, k)

    xm = x[-k:, :]  # xm contains the last k variables
    g = 1 + 9 / k * np.sum(xm, axis=0)

    # Compute the first M-1 objective functions
    fx = x[:M-1, :]

    # The last function requires another auxiliary variable
    gaux = g[np.newaxis, :]  # Replicate the g function
    h = M - np.sum(fx / (1 + gaux) * (1 + np.sin(3 * np.pi * fx)), axis=0)
    fx = (1 + g) * h

    return fx


import matplotlib.pyplot as plt
import seaborn as sns
if __name__ == "__main__":
    N = 100  # number of points
    x1 = np.linspace(0, 1, N)
    x2to21 = np.zeros((20, N))
    x = np.vstack((x1, x2to21))
    fx7 = dtlz7(x, 2)
    sns.lineplot(fx7, legend=False)
    plt.show()