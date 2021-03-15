"""
Test PCA.py

To test run `pytest` in the command line
"""
import numpy as np

from PCA import mean

def test_mean():
    assert(mean(1, 2, 3) == 2)
    assert(mean([1, 2, 3]) == 2)
    assert(np.array_equal(
        mean(
            np.array([1, 2, 3]),
            np.array([2, 3, 4]),
            np.array([3, 4, 5])
        ),
        np.array([2, 3, 4])
    ))
    assert(np.array_equal(
        mean([
            np.array([1, 2, 3]),
            np.array([2, 3, 4]),
            np.array([3, 4, 5])
        ]),
        np.array([2, 3, 4])
    ))
