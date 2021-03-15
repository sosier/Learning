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
