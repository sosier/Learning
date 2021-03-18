import numpy as np

def mean(*args):
    if len(args) == 1 and type(args[0]) == list:
        args = args[0]


    if isinstance(args[0], np.ndarray):
        total = np.zeros(shape=args[0].shape)
        for array in args:
            total += array
    else:
        total = sum(args)

    return total / len(args)

def variance(*args):
    avg = mean(*args)

    if len(args) == 1 and type(args[0]) == list:
        args = args[0]

    return mean([(val - avg)**2 for val in args])

def standard_deviation(*args):
    return np.sqrt(variance(*args))

def covariance(*args):
    """
    Covariance is only meaningful for when we have >1 dimension. As such the
    elements of args should be vectors of length > 1

    Returns the covariance matrix
    """
    avg = mean(*args)

    if len(args) == 1 and type(args[0]) == list:
        args = args[0]

    differences_vs_mean = [(val - avg) for val in args]
    differences_vs_mean = [
        np.expand_dims(diff, axis=1)
        if diff.ndim == 1
        else diff
        for diff in differences_vs_mean
    ]

    return mean([diff @ diff.T for diff in differences_vs_mean])

def correlation(*args):
    """
    Correlation is only meaningful for when we have >1 dimension. As such the
    elements of args should be vectors of length > 1

    Returns the correlation matrix
    """
    covariance_matrix = covariance(*args)

    standard_devs = np.expand_dims(standard_deviation(*args), axis=1)

    return covariance_matrix / (standard_devs @ standard_devs.T)
