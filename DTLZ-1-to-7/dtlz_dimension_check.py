import numpy as np

def dtlz_dimension_check(x, M, k):
    n = (M - 1) + k  # Required dimension
    if x.shape[0] != n:
        raise ValueError(f"Using k = {k}, we require the dimension size to be n = (M - 1) + k = {n} in this case.")

