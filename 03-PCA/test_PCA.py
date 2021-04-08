"""
Test PCA.py

To test run `pytest` in the command line
"""
import numpy as np

from PCA import (
    mean, variance, standard_deviation, covariance, correlation,
    define_inner_product, vector_length, vector_distance, vector_angle,
    project_vector_onto_subspace
)

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

def test_define_inner_product():
    assert(callable(define_inner_product()))
    assert(callable(define_inner_product("dot")))
    assert(callable(define_inner_product(np.array([[1, -1/2],
                                                   [-1/2, 1]]))))
def test_inner_product():
    IP = define_inner_product()
    assert(
        IP(np.array([1, 2]),
           np.array([1, 2]))
        == 5
    )
    assert(
        IP(np.array([1, 2]),
           np.array([2, 2]))
        == 6
    )
    assert(
        IP(np.array([[1],
                     [2]]),
           np.array([[1],
                     [2]]))
        == 5
    )
    assert(
        IP(np.array([[1],
                     [2]]),
           np.array([[2],
                     [2]]))
        == 6
    )

    IP = define_inner_product("dot")
    assert(
        IP(np.array([1, 2]),
           np.array([1, 2]))
        == 5
    )

    IP = define_inner_product(np.array([[2, 1, 0],
                                        [1, 2, -1],
                                        [0, -1, 2]]))
    assert(
        IP(np.array([1, -1, 3]),
           np.array([1, -1, 3]))
        == 26
    )
    assert(
        IP(np.array([1/2, -2, -1/2]),
           np.array([1/2, -2, -1/2]))
        == 5
    )
    assert(
        IP(np.array([4, 1, 1]),
           np.array([4, 1, 1]))
        == 42
    )

    IP = define_inner_product(np.array([[5/2, -1/2],
                                        [-1/2, 5/2]]))
    assert(
        IP(np.array([-1, 1]),
           np.array([-1, 1]))
        == 6
    )
    assert(
        IP(np.array([[-1],
                     [1]]),
           np.array([[-1],
                     [1]]))
        == 6
    )

def test_vector_length():
    IP = define_inner_product()
    assert(
        vector_length(np.array([1, 2]), IP)
        == np.sqrt(5)
    )

    IP = define_inner_product(np.array([[2, 1, 0],
                                        [1, 2, -1],
                                        [0, -1, 2]]))
    assert(
        vector_length(np.array([1, -1, 3]), IP)
        == np.sqrt(26)
    )
    assert(
        vector_length(np.array([1/2, -2, -1/2]), IP)
        == np.sqrt(5)
    )
    assert(
        vector_length(np.array([4, 1, 1]), IP)
        == np.sqrt(42)
    )

    IP = define_inner_product(np.array([[5/2, -1/2],
                                        [-1/2, 5/2]]))
    assert(
        vector_length(np.array([-1, 1]), IP)
        == np.sqrt(6)
    )

def test_vector_distance():
    IP = define_inner_product()
    assert(
        vector_distance(
            np.array([0, 4]),
            np.array([-1, 2]),
            IP
        ) == np.sqrt(5)
    )

    IP = define_inner_product(np.array([[2, 1, 0],
                                        [1, 2, -1],
                                        [0, -1, 2]]))
    assert(
        vector_distance(
            np.array([0, 4, 34]),
            np.array([-1, 5, 31]),
            IP
        ) == np.sqrt(26)
    )
    assert(
        vector_distance(
            np.array([0, 4, 3]),
            np.array([-1/2, 6, 3.5]),
            IP
        ) == np.sqrt(5)
    )
    assert(
        vector_distance(
            np.array([1, 11, 3.2]),
            np.array([-3, 10, 2.2]),
            IP
        ) == np.sqrt(42)
    )

    IP = define_inner_product(np.array([[5/2, -1/2],
                                        [-1/2, 5/2]]))
    assert(
        vector_distance(
            np.array([23, 3]),
            np.array([24, 2]),
            IP
        ) == np.sqrt(6)
    )

def test_vector_angle():
    IP = define_inner_product()
    assert(
        vector_angle(
            np.array([0, 4]),
            np.array([-1, 0]),
            IP
        ) == 90
    )
    assert(
        vector_angle(
            np.array([0, 4]),
            np.array([-1, 0]),
            IP,
            degrees=False
        ) == np.pi / 2
    )
    assert(np.isclose(
        vector_angle(
            np.array([0, 4]),
            np.array([1, 1]),
            IP
        ),
        45
    ))
    assert(np.isclose(
        vector_angle(
            np.array([0, 4]),
            np.array([1, 1]),
            IP,
            degrees=False
        ),
        np.pi / 4
    ))

    IP = define_inner_product(np.array([[2, 0],
                                        [0, 1]]))
    assert(
        vector_angle(
            np.array([1, 1]),
            np.array([-1, 1]),
            IP,
            degrees=False
        ) == np.arccos(-1/3)
    )

    IP = define_inner_product(np.array([[1, 0, 0],
                                        [0, 2, 0],
                                        [0, 0, 3]]))
    assert(
        vector_angle(
            np.array([2, 1, -4]),
            np.array([1, -1, 3]),
            IP,
            degrees=False
        ) == np.arccos(-2/np.sqrt(5))
    )

def test_project_vector_onto_subspace():
    assert(np.array_equal(
        np.round(
            project_vector_onto_subspace(
                np.array([6, 0, 0]),
                np.array([
                    [1, 0],
                    [1, 1],
                    [1, 2]
                ])
            ),
            4
        ),
        np.array([5., 2., -1.])
    ))
    assert(np.array_equal(
        project_vector_onto_subspace(
            np.array([3, 2, 2]),
            np.array([
                [1, 0],
                [0, 1],
                [0, 1]
            ])
        ),
        np.array([3, 2, 2])
    ))
    assert(np.array_equal(
        np.round(
            project_vector_onto_subspace(
                project_vector_onto_subspace(
                    np.array([12, 0, 0]),
                    np.array([
                        [1, 0],
                        [1, 1],
                        [1, 2]
                    ])
                ),
                np.array([
                    [-10 * np.sqrt(6)],
                    [-4 * np.sqrt(6)],
                    [2 * np.sqrt(6)]
                ])
            ),
            4
        ),
        np.array([10, 4, -2])
    ))
