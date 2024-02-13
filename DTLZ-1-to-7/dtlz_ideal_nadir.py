import numpy as np

def dtlz_ideal_nadir(fname, M):
    switch = {
        'dtlz1': (np.zeros(M), 0.5 * np.ones(M)),
        'dtlz2': (np.zeros(M), np.ones(M)),
        'dtlz3': (np.zeros(M), (1 / np.sqrt(2)) ** np.arange(M - 2, -1, -1)),
        'dtlz4': (np.zeros(M), np.ones(M)),
        'dtlz5': (np.zeros(M), (1 / np.sqrt(2)) ** np.arange(M - 2, -1, -1)),
        'dtlz6': (np.zeros(M), (1 / np.sqrt(2)) ** np.arange(M - 2, -1, -1)),
        'dtlz7': (np.concatenate((np.zeros(M - 1), [2 * M])), np.concatenate((np.ones(M - 1), [0.85940]))),
    }

    zstar, znad = switch.get(fname.lower(), (None, None))
    if zstar is None or znad is None:
        raise ValueError(f"Sorry, what the heck of a function is {fname}?")

    pareto_ranges = np.column_stack((zstar, znad))
    return pareto_ranges

