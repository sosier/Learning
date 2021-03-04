import numpy as np

def is_vector(object):
    return isinstance(object, np.ndarray) and object.ndim == 1

def is_matrix(object):
    return isinstance(object, np.ndarray) and object.ndim == 2

def relu(matrix):
    """
    RELU = REctified Linear Unit
    __/

    0 if x < 0 else x
    =
    max(0, x)
    """
    assert(is_matrix(matrix))
    return np.maximum(matrix, 0)

def derivative_of_relu(matrix):
    assert(is_matrix(matrix))

    derivative_for_zeroes = (
        (matrix == 0)  # 1. Find the zeros
        # 2. Randomly assign them 0 or 1:
        * np.random.random_sample(size=matrix.shape).round()
    )

    return (matrix > 0) + derivative_for_zeroes

def append_ones_column(matrix):
    assert(is_matrix(matrix))
    return np.append(matrix, np.ones((matrix.shape[0], 1)), axis=1)

def SimpleLayer(input_size=None, output_size=None, activation_function=relu,
                Wb="random"):
    """
    Returns a simple neural network layer function (e.g. Layer(...)) that can be
    called by passing in a valid input vector
    """
    assert(input_size is None or (type(input_size) == int and input_size >= 1))
    assert(output_size is None or (type(output_size) == int and output_size >= 1))
    assert(callable(activation_function))
    assert(is_matrix(Wb) or Wb == "random")

    # Allow either random initialization at a specified size OR Wb passed
    # in explicitly, but NOT both
    assert(
        (
            input_size is None and output_size is None
            and is_matrix(Wb)
        )
        or
        (
            input_size >= 1 and output_size >= 1
            and not is_matrix(Wb)
            and Wb == "random"
        )
    )

    if not is_matrix(Wb) and Wb == "random":
        # +1 to input size to combine W (weight) matrix with b (bias) vector:
        Wb = np.random.randn(input_size + 1, output_size)

    def Layer(X, return_all_internals=False):
        """
        Peforms a one layer neural network calculation:
            Z = X @ Wb
            Y = sigma(Z)
        where:
            Wb = The layer's weight + bias matrix
            X = The layer's input matrix, with samples as rows and features as
                columns
            Z = The intermediate value: X @ Wb
            sigma = The layer's activation function
            Y = The layer's output matrix
        """
        assert(is_matrix(X))
        assert((X.shape[1] + 1) == Wb.shape[0])

        # Append a column of 1 to multiple by the bias part of the Wb matrix:
        X = append_ones_column(X)

        if return_all_internals:
            Z = X @ Wb
            return {
                "Wb": Wb,
                "X": X,
                "Z": Z,
                "Y": activation_function(Z)
            }

        return activation_function(X @ Wb)

    return Layer

def derivative_of_SimpleLayer_Z(with_respect_to, X=None, Wb=None):
    """
    Derivative of Z = X @ Wb

    Supports all partial derivatives:
        dZ / dWb
        dZ / dX
    """
    assert(with_respect_to in ("Wb", "X"))

    if with_respect_to == "Wb":
        assert(not(X is None))
        return X
    elif with_respect_to == "X":
        assert(not(Wb is None))
        return Wb

def SimpleNeuralNetwork(*layers):
    """
    Returns a simple neural network function (e.g. NN(...)) that can be called
    by passing in an input matrix
    """
    if (len(layers) == 1
            and type(layers[0]) == list):
        layers = layers[0]

    assert(all(callable(layer) for layer in layers))

    def NN(X, return_all_internals=False):
        """
        Perform full "simple" / "dense" / "fully connected" neural network
        calculation, returning predictions

        X = Input data matrix
        """
        assert(is_matrix(X))
        result = X

        if return_all_internals:
            all_internals = []

            for layer in layers:
                internals = layer(result, return_all_internals=True)
                all_internals.append(internals)
                result = internals["Y"]

            return all_internals

        else:
            for layer in layers:
                result = layer(result)

            return result

    return NN

def mean_squared_error(predicted, actual):
    assert(predicted.shape == actual.shape)
    error = predicted - actual
    squared_error = error * error
    return np.expand_dims(  # Matrix
        # Summing all columns for each row and dividing by the number of columns
        np.sum(squared_error, axis=1) / squared_error.shape[1],
        axis=1
    )

def derivative_of_mean_squared_error(predicted, actual):
    assert(predicted.shape == actual.shape)
    return 2 * (predicted - actual)

def get_backprop_SimpleNeuralNetwork_jacobians(NN, X, Y):
    """
    NN = SimpleNeuralNetwork
    X = input matrix (where each row is a training example, and each column is a
        feature)
    Y = output / actuals matrix (where each row is a training example)
    """
    # Layer internals in reverse order (final layer first):
    all_internals = NN(X, return_all_internals=True)[::-1]

    jacobians = []  # J = Jacobians

    # Last layer's...
    # ...result a.k.a. Y:
    final_prediction = all_internals[0]["Y"]
    # ...intermediate_values a.k.a. Z:
    final_intermediate_values = all_internals[0]["Z"]

    derivative_of_cost = derivative_of_mean_squared_error(final_prediction, Y)
    derivative_of_final_layer_relu = derivative_of_relu(final_intermediate_values)

    for i in range(len(all_internals)):
        J = (
            derivative_of_cost
            * derivative_of_final_layer_relu
        )

        if i > 0:
            for j in range(i):
                J = (
                    J
                    @ np.delete(
                        derivative_of_SimpleLayer_Z(
                            with_respect_to="X",
                            Wb=all_internals[j]["Wb"],
                            X=all_internals[j]["X"]
                        ),
                        -1,  # Remove ("delete") bias row from the Wb matrix
                        axis=0
                    ).T  # Transpose so the resulting J is in the shape of the
                         # previous layer's Z (making the below multiplication
                         # possible)
                    * derivative_of_relu(all_internals[j+1]["Z"])
                )

        jacobians.append(
            (
                # Transpose J so that the matrix multiplication happens along
                # the axes that holds the training examples. This operation
                # performs both the multiplication part of of J * dZ/dWb and
                # summing portion of the average J across training examples
                J.T
                @ derivative_of_SimpleLayer_Z(
                    with_respect_to="Wb",
                    Wb=all_internals[i]["Wb"],
                    X=all_internals[i]["X"]
                )
            ).T  # Transpose back so the final J shape matches the shape of Wb
            / X.shape[0]  # Divide by the number of training examples to get the
                          # final average J across training examples
        )

    # Return jacobians in reverse order so that the first layer is first and the
    # final layer is last again:
    return jacobians[::-1]

def train_SimpleNeuralNetwork(NN, X, Y, learning_rate=0.001):
    """
    NN = SimpleNeuralNetwork
    X = input matrix (where each row is a training example, and each column is a
        feature)
    Y = output / actuals matrix (where each row is a training example)
    """
    all_internals = NN(X, return_all_internals=True)

    jacobians = get_backprop_SimpleNeuralNetwork_jacobians(NN, X, Y)
    jacobians = [learning_rate * J for J in jacobians]

    updated_Wb = [layer["Wb"] - J for layer, J in zip(all_internals, jacobians)]

    return SimpleNeuralNetwork([
        SimpleLayer(Wb=Wb)
        for Wb in updated_Wb
    ])

def find_root_Newton_Raphson(singe_variable_function, function_derivative,
                             starting_x=None, max_iterations=100):
    assert(callable(singe_variable_function))
    assert(callable(function_derivative))
    assert(max_iterations >= 0)

    # Starting values:
    if starting_x is None:
        x = np.random.random() * 20 - 10  # Random # between -10 & 10
    else:
        x = starting_x

    y = singe_variable_function(x)
    i = 0

    while i <= max_iterations:
        new_x = x - y / function_derivative(x)
        y = singe_variable_function(new_x)
        i += 1

        if y == 0 or x == new_x:
            return new_x
        else:
            x = new_x

    return np.nan
