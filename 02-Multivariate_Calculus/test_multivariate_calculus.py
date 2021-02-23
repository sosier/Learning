"""
Test multivariate_calculus.py

To test run `pytest` in the command line
"""
import numpy as np

from multivariate_calculus import (
    is_vector, is_matrix,
    relu, derivative_of_relu,
    append_ones_column,
    SimpleLayer, derivative_of_SimpleLayer_Z,
    SimpleNeuralNetwork,
    mean_squared_error, derivative_of_mean_squared_error,
    get_backprop_SimpleNeuralNetwork_jacobians
)

def test_is_vector():
    assert(is_vector([1, 2, 3]) is False)
    assert(is_vector(np.array([1, 2, 3])) is True)
    assert(is_vector(np.array([[1, 2, 3]])) is False)
    assert(is_vector(np.array([])) is True)

def test_is_matrix():
    assert(is_matrix([1, 2, 3]) is False)
    assert(is_matrix(np.array([1, 2, 3])) is False)
    assert(is_matrix(np.array([[1, 2, 3]])) is True)
    assert(is_matrix(np.array([[]])) is True)
    assert(is_matrix(np.array([
        [1],
        [1]
    ])) is True)

def test_relu():
    assert(relu(np.array([[-2]])) == np.array([[0]]))
    assert(relu(np.array([[-1000]])) == np.array([[0]]))
    assert(relu(np.array([[1000]])) == np.array([[1000]]))
    assert(relu(np.array([[0.1]])) == np.array([[0.1]]))
    assert(relu(np.array([[0]])) == np.array([[0]]))
    assert(np.array_equal(
        relu(np.array([[0, -1, 1]])),
        np.array([[0, 0, 1]])
    ))
    assert(np.array_equal(
        relu(np.array([[0],[-1], [1]])),
        np.array([[0], [0], [1]])
    ))
    assert(np.array_equal(
        relu(np.array([
            [0, -1, 1],
            [-1, 1, 2],
            [1, -7, -2]
        ])),
        np.array([
            [0, 0, 1],
            [0, 1, 2],
            [1, 0, 0]
        ])
    ))

def test_derivative_of_relu():
    assert(derivative_of_relu(np.array([[-2]])) == np.array([[0]]))
    assert(derivative_of_relu(np.array([[-1000]])) == np.array([[0]]))
    assert(derivative_of_relu(np.array([[1000]])) == np.array([[1]]))
    assert(derivative_of_relu(np.array([[0.1]])) == np.array([[1]]))
    assert(np.array_equal(
        derivative_of_relu(np.array([[0.1, -1, 1]])),
        np.array([[1, 0, 1]])
    ))
    assert(np.array_equal(
        derivative_of_relu(np.array([[0.1],[-1], [1]])),
        np.array([[1], [0], [1]])
    ))
    assert(np.array_equal(
        derivative_of_relu(np.array([
            [0.1, -1, 1],
            [-1, 1, 2],
            [1, -7, -2]
        ])),
        np.array([
            [1, 0, 1],
            [0, 1, 1],
            [1, 0, 0]
        ])
    ))

def test_append_ones_column():
    assert(np.array_equal(
        append_ones_column(np.array([[0]])),
        np.array([[0, 1]])
    ))
    assert(np.array_equal(
        append_ones_column(np.array([[0],
                                     [2]])),
        np.array([[0, 1],
                  [2, 1]])
    ))
    assert(np.array_equal(
        append_ones_column(np.array([[0, 1],
                                     [2, 3],
                                     [4, 5]])),
        np.array([[0, 1, 1],
                  [2, 3, 1],
                  [4, 5, 1]])
    ))

def test_SimpleLayer():
    assert(callable(SimpleLayer(input_size=1, output_size=1)))
    assert(is_matrix(
        SimpleLayer(input_size=1, output_size=1)(np.array([[0]]))
    ))
    assert(callable(
        SimpleLayer(
            Wb=np.array([[1, 2],
                         [1, 2]])
        )
    ))
    assert(is_matrix(
        SimpleLayer(
            Wb=np.array([[1, 2],
                         [1, 2]])
        )(np.array([[0]]))
    ))

    assert(np.array_equal(
        SimpleLayer(
            Wb=np.array([[1, 2],
                         [1, 2]])
        )(np.array([[0]])),
        # ==
        np.array([[1, 2]])
    ))
    assert(np.array_equal(
        SimpleLayer(
            Wb=np.array([[1, 2],
                         [-1, 2]])
        )(np.array([[0]])),
        # ==
        np.array([[0, 2]])
    ))
    assert(np.array_equal(
        SimpleLayer(
            Wb=np.array([[-1, 2],
                         [-1, 2]])
        )(np.array([[-2]])),
        # ==
        np.array([[1, 0]])
    ))

    all_internals = SimpleLayer(
        Wb=np.array([[-1, 2],
                     [-1, 2]])
    )(np.array([[-2]]), return_all_internals=True)
    assert(np.array_equal(
        all_internals["Wb"],
        # ==
        np.array([[-1, 2],
                  [-1, 2]])
    ))
    assert(np.array_equal(
        all_internals["X"],
        # ==
        np.array([[-2, 1]])
    ))
    assert(np.array_equal(
        all_internals["Z"],
        # ==
        np.array([[1, -2]])
    ))
    assert(np.array_equal(
        all_internals["Y"],
        # ==
        np.array([[1, 0]])
    ))

def test_derivative_of_SimpleLayer_Z():
    assert(np.array_equal(
        derivative_of_SimpleLayer_Z(
            with_respect_to="Wb",
            Wb=np.array([[-1, 2],
                         [-1, 2]]),
            X=np.array([[-2, 1]])
        ),
        # ==
        np.array([[-2, 1]])
    ))
    assert(np.array_equal(
        derivative_of_SimpleLayer_Z(
            with_respect_to="X",
            Wb=np.array([[-1, 2],
                         [-1, 2]]),
            X=np.array([[-2, 1]])
        ),
        # ==
        np.array([[-1, 2],
                  [-1, 2]])
    ))

def test_SimpleNeuralNetwork():
    assert(callable(
        SimpleNeuralNetwork(
            SimpleLayer(input_size=2, output_size=2)
        )
    ))
    assert(callable(
        SimpleNeuralNetwork(
            SimpleLayer(input_size=3, output_size=2),
            SimpleLayer(input_size=2, output_size=1)
        )
    ))
    assert(callable(
        SimpleNeuralNetwork([
            SimpleLayer(input_size=3, output_size=2),
            SimpleLayer(input_size=2, output_size=1)
        ])
    ))
    assert(is_matrix(
        SimpleNeuralNetwork(
            SimpleLayer(input_size=2, output_size=2)
        )(np.array([[1, 1]]))
    ))
    assert(is_matrix(
        SimpleNeuralNetwork(
            SimpleLayer(input_size=3, output_size=2),
            SimpleLayer(input_size=2, output_size=1)
        )(np.array([[1, 1, 1]]))
    ))
    assert(is_matrix(
        SimpleNeuralNetwork([
            SimpleLayer(input_size=3, output_size=2),
            SimpleLayer(input_size=2, output_size=1)
        ])(np.array([[1, 1, 1]]))
    ))

    assert(callable(
        SimpleNeuralNetwork(
            SimpleLayer(Wb=np.array([[1, 2],
                                     [1, 2]]))
        )
    ))
    assert(is_matrix(
        SimpleNeuralNetwork(
            SimpleLayer(Wb=np.array([[1, 2],
                                     [1, 2]]))
        )(np.array([[0]]))
    ))
    assert(np.array_equal(
        SimpleNeuralNetwork(
            SimpleLayer(Wb=np.array([[1, 2],
                                     [1, 2]]))
        )(np.array([[0]])),
        # ==
        np.array([[1, 2]])
    ))
    assert(np.array_equal(
        SimpleNeuralNetwork(
            SimpleLayer(Wb=np.array([[1, 2],
                                     [1, 2]])),
            SimpleLayer(Wb=np.array([[1, 1],
                                     [2, 2],
                                     [1, 2]]))
        )(np.array([[0]])),
        # ==
        np.array([[6, 7]])
    ))

    all_internals = SimpleNeuralNetwork(
        SimpleLayer(Wb=np.array([[1, 2],
                                 [1, 2]])),
        SimpleLayer(Wb=np.array([[1, 1],
                                 [2, 2],
                                 [1, 2]]))
    )(np.array([[0]]), return_all_internals=True)

    # Layer 1:
    assert(np.array_equal(
        all_internals[0]["Wb"],
        # ==
        np.array([[1, 2],
                  [1, 2]])
    ))
    assert(np.array_equal(
        all_internals[0]["X"],
        # ==
        np.array([[0, 1]])
    ))
    assert(np.array_equal(
        all_internals[0]["Z"],
        # ==
        np.array([[1, 2]])
    ))
    assert(np.array_equal(
        all_internals[0]["Y"],
        # ==
        np.array([[1, 2]])
    ))
    # Layer 2:
    assert(np.array_equal(
        all_internals[1]["Wb"],
        # ==
        np.array([[1, 1],
                  [2, 2],
                  [1, 2]])
    ))
    assert(np.array_equal(
        all_internals[1]["X"],
        # ==
        np.array([[1, 2, 1]])
    ))
    assert(np.array_equal(
        all_internals[1]["Z"],
        # ==
        np.array([[6, 7]])
    ))
    assert(np.array_equal(
        all_internals[1]["Y"],
        # ==
        np.array([[6, 7]])
    ))

def test_mean_squared_error():
    assert(np.array_equal(
        mean_squared_error(
            np.array([[2]]),
            np.array([[4]])
        ),
        # ==
        np.array([[4]])
    ))
    assert(np.array_equal(
        mean_squared_error(
            np.array([[2]]),
            np.array([[3]])
        ),
        # ==
        np.array([[1]])
    ))
    assert(np.array_equal(
        mean_squared_error(
            np.array([[3]]),
            np.array([[1]])
        ),
        # ==
        np.array([[4]])
    ))
    assert(np.array_equal(
        mean_squared_error(
            np.array([[2, 2]]),
            np.array([[3, 4]])
        ),
        # ==
        np.array([[2.5]])
    ))
    assert(np.array_equal(
        mean_squared_error(
            np.array([[2],
                      [2]]),
            np.array([[3],
                      [4]])
        ),
        # ==
        np.array([[1],
                  [4]])
    ))
    assert(np.array_equal(
        mean_squared_error(
            np.array([[2, 3],
                      [2, 3]]),
            np.array([[3, 3],
                      [4, 4]])
        ),
        # ==
        np.array([[0.5],
                  [2.5]])
    ))

def test_derivative_of_mean_squared_error():
    assert(np.array_equal(
        derivative_of_mean_squared_error(
            np.array([[2]]),
            np.array([[4]])
        ),
        # ==
        np.array([[-4]])
    ))
    assert(np.array_equal(
        derivative_of_mean_squared_error(
            np.array([[2]]),
            np.array([[3]])
        ),
        # ==
        np.array([[-2]])
    ))
    assert(np.array_equal(
        derivative_of_mean_squared_error(
            np.array([[3]]),
            np.array([[1]])
        ),
        # ==
        np.array([[4]])
    ))
    assert(np.array_equal(
        derivative_of_mean_squared_error(
            np.array([[2, 2]]),
            np.array([[3, 4]])
        ),
        # ==
        np.array([[-2, -4]])
    ))
    assert(np.array_equal(
        derivative_of_mean_squared_error(
            np.array([[2],
                      [2]]),
            np.array([[3],
                      [4]])
        ),
        # ==
        np.array([[-2],
                  [-4]])
    ))
    assert(np.array_equal(
        derivative_of_mean_squared_error(
            np.array([[2, 3],
                      [2, 3]]),
            np.array([[3, 3],
                      [4, 4]])
        ),
        # ==
        np.array([[-2, 0],
                  [-4, -2]])
    ))

def test_get_backprop_SimpleNeuralNetwork_jacobians():
    assert(
        get_backprop_SimpleNeuralNetwork_jacobians(
            SimpleNeuralNetwork(
                SimpleLayer(input_size=2, output_size=3),
                SimpleLayer(input_size=3, output_size=1),
            ),
            X=np.array([[1, 2], [3, 4]]),
            Y=np.array([[5], [6]])
        )[0].shape == (3, 3)
    )
    assert(
        get_backprop_SimpleNeuralNetwork_jacobians(
            SimpleNeuralNetwork(
                SimpleLayer(input_size=2, output_size=3),
                SimpleLayer(input_size=3, output_size=1),
            ),
            X=np.array([[1, 2], [3, 4]]),
            Y=np.array([[5], [6]])
        )[1].shape == (4, 1)
    )
    assert(
        get_backprop_SimpleNeuralNetwork_jacobians(
            SimpleNeuralNetwork(
                SimpleLayer(input_size=2, output_size=12),
                SimpleLayer(input_size=12, output_size=1),
            ),
            X=np.array([[1, 2], [3, 4]]),
            Y=np.array([[5], [6]])
        )[0].shape == (3, 12)
    )
    assert(
        get_backprop_SimpleNeuralNetwork_jacobians(
            SimpleNeuralNetwork(
                SimpleLayer(input_size=2, output_size=12),
                SimpleLayer(input_size=12, output_size=1),
            ),
            X=np.array([[1, 2], [3, 4]]),
            Y=np.array([[5], [6]])
        )[1].shape == (13, 1)
    )
    assert(
        get_backprop_SimpleNeuralNetwork_jacobians(
            SimpleNeuralNetwork(
                SimpleLayer(input_size=2, output_size=12),
                SimpleLayer(input_size=12, output_size=1),
            ),
            X=np.array([[1, 2], [3, 4]]),
            Y=np.array([[5], [6]])
        )[0].shape == (3, 12)
    )
    assert(
        get_backprop_SimpleNeuralNetwork_jacobians(
            SimpleNeuralNetwork(
                SimpleLayer(input_size=2, output_size=12),
                SimpleLayer(input_size=12, output_size=5),
                SimpleLayer(input_size=5, output_size=1),
            ),
            X=np.array([[1, 2], [3, 4]]),
            Y=np.array([[5], [6]])
        )[0].shape == (3, 12)
    )
    assert(
        get_backprop_SimpleNeuralNetwork_jacobians(
            SimpleNeuralNetwork(
                SimpleLayer(input_size=2, output_size=12),
                SimpleLayer(input_size=12, output_size=5),
                SimpleLayer(input_size=5, output_size=1),
            ),
            X=np.array([[1, 2], [3, 4]]),
            Y=np.array([[5], [6]])
        )[1].shape == (13, 5)
    )
    assert(
        get_backprop_SimpleNeuralNetwork_jacobians(
            SimpleNeuralNetwork(
                SimpleLayer(input_size=2, output_size=12),
                SimpleLayer(input_size=12, output_size=5),
                SimpleLayer(input_size=5, output_size=1),
            ),
            X=np.array([[1, 2], [3, 4]]),
            Y=np.array([[5], [6]])
        )[2].shape == (6, 1)
    )
