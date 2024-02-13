import numpy as np

def dtlz_range(fname, M):
    switch = {
        'dtlz1': 5,
        'dtlz2': 10,
        'dtlz3': 10,
        'dtlz4': 10,
        'dtlz5': 10,
        'dtlz6': 10,
        'dtlz7': 20,
    }

    k = switch.get(fname.lower())
    if k is None:
        raise ValueError(f"Sorry, what the heck of a function is {fname}?")

    n = (M - 1) + k  # number of decision variables
    xlims = np.column_stack((np.zeros(n), np.ones(n)))
    return xlims
