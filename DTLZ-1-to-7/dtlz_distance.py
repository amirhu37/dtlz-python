import numpy as np

def dtlz_distance(fname, xopt):
    # For each function, a different value of k and reference point is given
    fname = fname.lower()
    if fname == 'dtlz1':
        k = 5
        value = 0.5
    elif fname in ['dtlz2', 'dtlz3', 'dtlz4', 'dtlz5']:
        k = 10
        value = 0.5
    elif fname == 'dtlz6':
        k = 10
        value = 0
    elif fname == 'dtlz7':
        k = 20
        value = 0
    else:
        raise ValueError(f"Sorry, what the heck of a function is {fname}?")

    mu = xopt.shape[1]  # number of points
    xlast = np.tile(value, (k, mu))  # replicate the required value
    d = np.sum((xopt[-k:, :] - xlast) ** 2, axis=0)
    return d

# Example usage:
if __name__ == "__main__":
    # Create a sample input matrix xopt (replace with your actual data)
    xopt = np.random.rand(10, 5)  # Example: 10 points with dimension 5
    function_name = 'dtlz1'  # Replace with the desired DTLZ function name

    try:
        distances = dtlz_distance(function_name, xopt)
        print("Distances:", distances)
    except ValueError as e:
        print(f"Error: {e}")
