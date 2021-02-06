import traceback

from multivariate_calculus import (
    is_vector, is_matrix, relu, SimpleLayer, SimpleNeuralNetwork
)
from linear_algebra import Vector, Matrix

PRECISION = 10

def test_is_vector():
    try:
        assert(is_vector([1, 2, 3]) is False)
        assert(is_vector(Vector([1, 2, 3])) is True)
        assert(is_vector(Matrix([[1, 2, 3]])) is False)
        assert(is_vector(Vector([])) is True)

        print("All is_vector tests pass")
        return True
    except Exception:
        print("is_vector TEST FAILED!")
        traceback.print_exc()
        return False

def test_is_matrix():
    try:
        assert(is_matrix([1, 2, 3]) is False)
        assert(is_matrix(Vector([1, 2, 3])) is False)
        assert(is_matrix(Matrix([[1, 2, 3]])) is True)
        assert(is_matrix(Matrix([])) is True)
        assert(is_matrix(Matrix([
            [1],
            [1]
        ])) is True)

        print("All is_matrix tests pass")
        return True
    except Exception:
        print("is_matrix TEST FAILED!")
        traceback.print_exc()
        return False

def test_relu():
    try:
        assert(relu(-2) == 0)
        assert(relu(-1000) == 0)
        assert(relu(1000) == 1000)
        assert(relu(0.1) == 0.1)
        assert(relu(0) == 0)
        assert(relu(Vector(0, -1, 1)) == Vector(0, 0, 1))
        assert(relu(Matrix([0, -1, 1])) == Matrix([0, 0, 1]))

        print("All relu tests pass")
        return True
    except Exception:
        print("relu TEST FAILED!")
        traceback.print_exc()
        return False

def test_SimpleLayer():
    try:
        assert(callable(SimpleLayer(input_size=1, output_size=1)))
        assert(is_vector(
            SimpleLayer(input_size=1, output_size=1)(Vector([0]))
        ))
        assert(callable(SimpleLayer(W=Matrix([1, 2]), b=Vector(1, 2))))
        assert(is_vector(
            SimpleLayer(W=Matrix([[1], [2]]), b=Vector(1, 2))(Vector([0]))
        ))
        assert(
            SimpleLayer(W=Matrix([[1], [2]]), b=Vector(1, 2))(Vector([0]))
            == Vector([1, 2])
        )
        assert(
            SimpleLayer(W=Matrix([[1], [2]]), b=Vector(-1, 2))(Vector([0]))
            == Vector([0, 2])
        )
        assert(
            SimpleLayer(W=Matrix([[-1], [2]]), b=Vector(-1, 2))(Vector([-2]))
            == Vector([1, 0])
        )

        print("All SimpleLayer tests pass")
        return True
    except Exception:
        print("SimpleLayer TEST FAILED!")
        traceback.print_exc()
        return False

def test_SimpleNeuralNetwork():
    try:
        assert(callable(
            SimpleNeuralNetwork(SimpleLayer(input_size=2, output_size=2))
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
        assert(is_vector(
            SimpleNeuralNetwork(
                SimpleLayer(input_size=2, output_size=2)
            )(Vector([1, 1]))
        ))
        assert(is_vector(
            SimpleNeuralNetwork(
                SimpleLayer(input_size=3, output_size=2),
                SimpleLayer(input_size=2, output_size=1)
            )(Vector([1, 1, 1]))
        ))
        assert(is_vector(
            SimpleNeuralNetwork([
                SimpleLayer(input_size=3, output_size=2),
                SimpleLayer(input_size=2, output_size=1)
            ])(Vector([1, 1, 1]))
        ))

        assert(callable(
            SimpleNeuralNetwork(
                SimpleLayer(W=Matrix([[1], [2]]), b=Vector(1, 2))
            )
        ))
        assert(is_vector(
            SimpleNeuralNetwork(
                SimpleLayer(W=Matrix([[1], [2]]), b=Vector(1, 2))
            )(Vector([0]))
        ))
        assert(
            SimpleNeuralNetwork(
                SimpleLayer(W=Matrix([[1], [2]]), b=Vector(1, 2))
            )(Vector([0]))
            == Vector([1, 2])
        )
        assert(
            SimpleNeuralNetwork(
                SimpleLayer(W=Matrix([[1], [2]]), b=Vector(1, 2)),
                SimpleLayer(W=Matrix([[1, 1], [2, 2]]), b=Vector(1, 2))
            )(Vector([0]))
            == Vector([4, 8])
        )

        print("All SimpleNeuralNetwork tests pass")
        return True
    except Exception:
        print("SimpleNeuralNetwork TEST FAILED!")
        traceback.print_exc()
        return False

test_is_vector()
test_is_matrix()
test_relu()
test_SimpleLayer()
test_SimpleNeuralNetwork()
