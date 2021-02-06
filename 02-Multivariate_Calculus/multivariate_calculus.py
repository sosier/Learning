import sys
import random

sys.path.append('../01-Linear_Algebra/')
from linear_algebra import (
    Vector, Matrix, is_numeric
)

def is_vector(object):
    return isinstance(object, Vector)

def is_matrix(object):
    return isinstance(object, Matrix)

def relu(object):
    """
    RELU = REctified Linear Unit
    __/

    0 if x < 0 else x
    =
    max(0, x)
    """
    if is_numeric(object):
        return max(0, object)

    if is_vector(object):
        return Vector([relu(val) for val in object.vector])

    if is_matrix(object):
        return Matrix([
            [relu(val) for val in row]
            for row in object.matrix
        ])

def SimpleLayer(input_size=None, output_size=None, activation_function=relu,
                W="random", b="random"):
    """
    Returns a simple neural network layer function (e.g. Layer(...)) that can be
    called by passing in a valid input vector
    """
    assert(input_size is None or (type(input_size) == int and input_size >= 1))
    assert(output_size is None or (type(output_size) == int and output_size >= 1))
    assert(callable(activation_function))
    assert(is_matrix(W) or W == "random")
    assert(is_vector(b) or b == "random")

    # Allow either random initialization at a specified size OR W and b passed
    # in explicitly, but NOT both
    assert(
        (
            input_size is None and output_size is None
            and is_matrix(W) and is_vector(b)
        )
        or
        (
            input_size >= 1 and output_size >= 1
            and W == "random" and b == "random"
        )
    )

    if W == "random" and b == "random":
        W = Matrix([
            [random.uniform(-1, 1) for c in range(input_size)]
            for r in range(output_size)
        ])
        b = Vector([random.uniform(-1, 1) for _ in range(output_size)])

    def Layer(input_vector):
        assert(is_vector(input_vector))
        assert(len(input_vector) == W.num_columns)
        return activation_function(W * input_vector + b)

    return Layer

def SimpleNeuralNetwork(*layers):
    """
    Returns a simple neural network function (e.g. NN(...)) that can be called
    by passing in an input vector
    """
    if (len(layers) == 1
            and type(layers[0]) == list):
        layers = layers[0]

    assert(all(callable(layer) for layer in layers))

    def NN(input_vector):
        assert(is_vector(input_vector))
        result = input_vector

        for layer in layers:
            result = layer(result)

        return result

    return NN
