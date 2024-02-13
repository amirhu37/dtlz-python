from dtlz_dimension_check import dtlz_dimension_check
import numpy as np

def dtlz6(x, M):
    k = 10

    dtlz_dimension_check(x,M,k)

    xm = x[-k:, :]  # xm contains the last k variables
    g = np.sum(xm ** 0.1, axis=0)

    theta = np.empty((M, x.shape[1]))
    theta[0, :] = np.pi / 2 * x[0, :]
    gr = g[np.newaxis, :]  # Replicate gr for the multiplication below
    theta[1:M-1, :] = np.pi / (4 * (1 + gr)) * (1 + 2 * gr * x[1:M-1, :])

    # Compute the functions
    fx = np.empty((M, x.shape[1]))
    fx[0, :] = (1 + g) * np.prod(np.cos(theta[:M-1, :]), axis=0)
    for ii in range(1, M - 1):
        fx[ii, :] = (1 + g) * np.prod(np.cos(theta[:M-ii, :]), axis=0) * \
                    np.sin(theta[M - ii, :])
    fx[-1, :] = (1 + g) * np.sin(theta[0, :])

    return fx



import matplotlib.pyplot as plt
import seaborn as sns
if __name__ == "__main__":
    N = 20  # The actual number of solutions will be N^2
    xrange = np.linspace(0, 1, N)
    x1to2 = np.zeros((2, 0))

    for i in xrange:
        x1to2 = np.hstack((x1to2, np.vstack((i * np.ones((1,N)), xrange))))

    x5to12 = np.zeros((10, N**2))
    x = np.vstack((x1to2, x5to12))

    fx6 = dtlz6(x, 3)
    sns.lineplot(fx6, legend=False)
    plt.show()