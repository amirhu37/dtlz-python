import numpy as np

def ndset(F : np.ndarray):
    mu = F.shape[1]  # Number of points

    # Compare each point with the others
    f1 = np.transpose(F, (0, 2, 1))  # Puts in the 3D direction
    f1 = np.tile(f1, (1, mu, 1))
    f2 = np.tile(F, (1, 1, mu))

    # Check where f1 dominates f2
    aux1 = np.all(f1 <= f2, axis=2)
    aux2 = np.any(f1 < f2, axis=2)
    auxf1 = np.logical_and(aux1, aux2)

    # Check where f1 is dominated by f2
    aux1 = np.all(f1 >= f2, axis=2)
    aux2 = np.any(f1 > f2, axis=2)
    auxf2 = np.logical_and(aux1, aux2)

    # dom will be a 3D matrix (1 x mu x mu) such that, for the ii-th slice,
    # it will contain +1 if fii dominates the current point, -1 if it is dominated
    # by it, and 0 if they are incomparable
    dom = np.zeros((1, mu, mu))
    dom[auxf1] = 1
    dom[auxf2] = -1

    # Finally, the slices with no -1 are nondominated
    ispar = np.all(dom != -1, axis=2)
    ispar = ispar.flatten()

    return ispar
