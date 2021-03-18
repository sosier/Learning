"""
Test PCA.py

To test run `pytest` in the command line
"""
import numpy as np

from PCA import mean, variance, standard_deviation, covariance, correlation

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

def test_variance():
    assert(variance(1, 2, 3) == 2/3)
    assert(variance([1, 2, 3]) == 2/3)
    assert(variance(0, 2, 4) == 8/3)
    assert(variance(600, 470, 170, 430, 300) == 21704)
    assert(np.array_equal(
        variance(
            np.array([1, 1, 0]),
            np.array([2, 2, 2]),
            np.array([3, 3, 4])
        ),
        np.array([2/3, 2/3, 8/3])
    ))
    assert(np.array_equal(
        variance([
            np.array([1, 1, 0]),
            np.array([2, 2, 2]),
            np.array([3, 3, 4])
        ]),
        np.array([2/3, 2/3, 8/3])
    ))

def test_standard_deviation():
    assert(standard_deviation(1, 3) == 1)
    assert(standard_deviation([1, 3]) == 1)
    assert(standard_deviation(0, 4) == 2)
    assert(standard_deviation(600, 470, 170, 430, 300) == np.sqrt(21704))
    assert(np.array_equal(
        standard_deviation(
            np.array([1, 1, 0]),
            np.array([3, 3, 4])
        ),
        np.array([1, 1, 2])
    ))
    assert(np.array_equal(
        standard_deviation([
            np.array([1, 1, 0]),
            np.array([3, 3, 4])
        ]),
        np.array([1, 1, 2])
    ))

def test_covariance():
    assert(np.array_equal(
        covariance(
            np.array([1, 2]),
            np.array([5, 4])
        ),
        np.array([
            [4, 2],
            [2, 1]
        ])
    ))
    assert(np.array_equal(
        covariance([
            np.array([1, 2]),
            np.array([5, 4])
        ]),
        np.array([
            [4, 2],
            [2, 1]
        ])
    ))
    assert(np.array_equal(
        covariance(
            np.array([1, 2]),
            np.array([7, 4])
        ),
        np.array([
            [9, 3],
            [3, 1]
        ])
    ))

def test_correlation():
    assert(np.array_equal(
        correlation(
            np.array([1, 2]),
            np.array([2, 4])
        ),
        np.array([
            [1, 1],
            [1, 1]
        ])
    ))
    assert(np.array_equal(
        correlation([
            np.array([1, 2]),
            np.array([2, 4])
        ]),
        np.array([
            [1, 1],
            [1, 1]
        ])
    ))
    assert(np.array_equal(
        correlation(
            np.array([1, 2]),
            np.array([2, -4])
        ),
        np.array([
            [1, -1],
            [-1, 1]
        ])
    ))
    assert(np.array_equal(
        correlation(
            np.array([1, 2]),
            np.array([1, -2]),
            np.array([2, 2]),
            np.array([2, -2])
        ),
        np.array([
            [1, 0],
            [0, 1]
        ])
    ))
    assert(np.array_equal(
        np.round(correlation(
            np.array([14.2, 215]),
            np.array([16.4, 325]),
            np.array([11.9, 185]),
            np.array([15.2, 332]),
            np.array([18.5, 406]),
            np.array([22.1, 522]),
            np.array([19.4, 412]),
            np.array([25.1, 614]),
            np.array([23.4, 544]),
            np.array([18.1, 421]),
            np.array([22.6, 445]),
            np.array([17.2, 408])
        ), 4),
        np.array([
            [1, 0.9575],
            [0.9575, 1]
        ])
    ))
    assert(np.array_equal(
        np.round(correlation(
            np.array([14.2, -215]),
            np.array([16.4, -325]),
            np.array([11.9, -185]),
            np.array([15.2, -332]),
            np.array([18.5, -406]),
            np.array([22.1, -522]),
            np.array([19.4, -412]),
            np.array([25.1, -614]),
            np.array([23.4, -544]),
            np.array([18.1, -421]),
            np.array([22.6, -445]),
            np.array([17.2, -408])
        ), 4),
        np.array([
            [1, -0.9575],
            [-0.9575, 1]
        ])
    ))
