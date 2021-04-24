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

def define_inner_product(matrix="dot"):
    if type(matrix) == str and matrix == "dot":
        def inner_product(vector_x, vector_y):
            x = np.squeeze(vector_x) if vector_x.ndim >= 2 else vector_x
            y = np.squeeze(vector_y) if vector_y.ndim >= 2 else vector_y
            return np.dot(x, y)

    else:
        def inner_product(vector_x, vector_y):
            x = np.expand_dims(vector_x, axis=1) if vector_x.ndim == 1 else vector_x
            y = np.expand_dims(vector_y, axis=1) if vector_y.ndim == 1 else vector_y

            return x.T @ matrix @ y

    return inner_product

def vector_length(vector, inner_product):
    return np.sqrt(inner_product(vector, vector))

def vector_distance(vector_x, vector_y, inner_product):
    return vector_length(vector_x - vector_y, inner_product)

def vector_angle(vector_x, vector_y, inner_product, degrees=True):
    cosine_of_angle = (
        inner_product(vector_x, vector_y)
        / (
            vector_length(vector_x, inner_product)
            * vector_length(vector_y, inner_product)
        )
    )

    angle = np.arccos(cosine_of_angle)

    if degrees:
        return angle / (2 * np.pi) * 360
    else:
        return angle

def project_vector_onto_subspace(vector, subspace_bases_matrix):
    """
    Assumes inner product = the dot product
    """
    B = subspace_bases_matrix  # Columns = The bases
    B = np.expand_dims(B, axis=1) if B.ndim == 1 else B

    projection_matrix = B @ np.linalg.inv(B.T @ B) @ B.T

    return projection_matrix @ vector

def PCA(data, n_components):
    """
    data = Matrix (N x D) with rows as observations and columns as features
    new_dimensions
    """
    N, D = data.shape
    assert(n_components <= D)

    # 1. Center the data (convert features to having mean 0)
    means = np.mean(data, axis=0)
    centered_data = data - means

    # 2. Calculate data covariance_matrix
    covariance_matrix = (centered_data.T @ centered_data) / N

    # 3. Calculate eigen- values / vectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # 4. Order these from largest to smallest, and select the largest
    #    `new_dimensions` number of vectors. These are the principle components
    sort_order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_order]
    eigenvectors = eigenvectors[:, sort_order]
    eigenvalues = eigenvalues[:n_components]
    eigenvectors = eigenvectors[:, :n_components]

    # print(eigenvalues)

    # 5. Project the centered data onto the principle components
    return centered_data @ eigenvectors

# x = np.array([[1, 2, 3]])
# print(x)
# print(x.ndim)
# x = np.expand_dims(x, axis=1)
# print(x)
# print(x.ndim)
