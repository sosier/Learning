from math import sqrt, acos, degrees
from itertools import combinations

def is_numeric(object):
    return type(object) == int or type(object) == float

class Vector():
    """
    A vector is an ordered, list of numbers
    """
    def __init__(self, *numbers):
        if len(numbers) == 1 and type(numbers[0]) == list:
            numbers = numbers[0]

        assert(all(is_numeric(num) for num in numbers))
        self.vector = list(numbers)

    def __str__(self):
        return str(self.vector)

    def __len__(self):
        """
        This is actually more like the "dimension" of the vector, but using the
        Python `len` command will return the `len` the same way it would for a
        list of numbers
        """
        return len(self.vector)

    def __eq__(self, other):
        try:
            return (
                len(self) == len(other)
                and all(x == y for x, y in zip(self, other))
            )
        except:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getitem__(self, i):
        return self.vector[i]

    def __setitem__(self, i, value):
        self.vector[i] = value

    def __add__(self, other):
        assert(isinstance(other, Vector))
        assert(len(self) == len(other))
        return Vector(*[x + y for x, y in zip(self, other)])

    def __sub__(self, other):
        assert(isinstance(other, Vector))
        assert(len(self) == len(other))
        return Vector(*[x - y for x, y in zip(self, other)])

    def __mul__(self, other):
        assert(is_numeric(other))
        return Vector(*[other * num for num in self.vector])

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        assert(is_numeric(other))
        return Vector(*[1/other * num for num in self.vector])

    def __round__(self, ndigits=None):
        return Vector(*[round(num, ndigits) for num in self.vector])

    def magnitude(self):
        """
        "Magnitude", "size", or "length" of the vector.

        Calculated via the Pythagorean thereom. In that way, this is also the
        distance from the origin to the end of the vector.
        """
        return sqrt(sum([num**2 for num in self.vector]))

    def dot_product(self, other):
        """
        "Dot product" or "sum product" of two vectors
        """
        assert(isinstance(other, Vector))
        assert(len(self) == len(other))
        return sum([x * y for x, y in zip(self, other)])

    def dot(self, other):
        """
        Short convenience function for accessing `.dot_product(...)` since this
        is such a common vector operation
        """
        return self.dot_product(other)

    def cosine_similarity(self, other):
        return (
            self.dot(other) /
            (self.magnitude() * other.magnitude())
        )

    def cosine_distance(self, other):
        """
        Convert the cosine similarity between two vectors into a distance metric
        """
        return 1 - self.cosine_similarity(other)

    def angle_between(self, other, radians=False):
        angle = acos(self.cosine_similarity(other))
        if not radians:
            angle = degrees(angle)

        return angle

    def orthogonal_to(self, other):
        return self.dot(other) == 0

    def scalar_projection_onto(self, other):
        """
        Given vectors a & b, with angle theta between them:
         - Knowing:
           - cosine = cos(angle) = adjacent / hypotenuse
           - cos(theta) = cosine_similarity = a . b / (||a|| * ||b||)
         - Projection of a onto b:
            1. ||a|| = hypotenuse
            2. Projection of a onto b = adjacent
            3. cos(theta) = adjacent / hypotenuse = Projection of a onto b / ||a||
            4. Projection of a onto b = ||a|| * cos(theta)
            5. Projection of a onto b = ||a|| * a . b / (||a|| * ||b||)
            6. Projection of a onto b = a . b / ||b||

        Note, the scalar projection is the length / magnitude of the projection,
        i.e. how many unit vectors in the direction of b
        """
        return (
            self.dot(other) /
            other.magnitude()
        )

    def normalize(self):
        """
        Convert vector to a vector of length one in the same direction
        """
        return self / self.magnitude()

    def vector_projection_onto(self, other):
        """
        The projection of `self` onto `other` into vector form
        """
        return self.scalar_projection_onto(other) * other.normalize()

    def vector_projection_scalar(self, other):
        """
        The scalar amount by which the `other` vector must be scaled (multiplied)
        to yield the vector projection
        """
        return self.scalar_projection_onto(other) / other.magnitude()

    def change_basis(self, *basis_vectors):
        assert(all([isinstance(vector, Vector) for vector in basis_vectors]))
        assert(all([len(self) == len(vector) for vector in basis_vectors]))
        assert(all([
            # All basis vectors must be orthogonal to eachother
            vector_a.orthogonal_to(vector_b)
            for vector_a, vector_b in combinations(basis_vectors, 2)
        ]))
        return Vector(*[
            self.vector_projection_scalar(basis_vector)
            for basis_vector in basis_vectors
        ])

class Matrix():
    """
    A matrix is an ordered, "rectangular" list of numbers (with rows and columns)
    """
    def __init__(self, *rows_of_numbers):
        if (len(rows_of_numbers) == 1
                and type(rows_of_numbers[0]) == list
                and type(rows_of_numbers[0][0]) == list):
            rows_of_numbers = rows_of_numbers[0]

        for row in rows_of_numbers:
            assert(type(row) == list)
            assert(all(is_numeric(num) for num in row))
            assert(len(row) == len(rows_of_numbers[0]))

        self.matrix = list(rows_of_numbers)

    def __str__(self):
        return "\n".join([str(row) for row in self.matrix]) + "\n"

    def dimensions(self):
        """
        Return the matrix dimensions (m x n) / (rows x columns) as a tuple:
        (m, n) / (rows, columns)
        """
        return (
            len(self.matrix),  # Rows
            len(self.matrix[0])  # Columns
        )

    def __eq__(self, other):
        try:
            return (
                self.dimensions() == other.dimensions()
                and all(
                    self_val == other_val
                    for self_row, other_row in zip(self.matrix, other.matrix)
                    for self_val, other_val in zip(self_row, other_row)
                )
            )
        except:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getitem__(self, i):
        return self.matrix[i]

    def __setitem__(self, i, value):
        self.matrix[i] = value

    def __round__(self, ndigits=None):
        return Matrix([
            [round(num, ndigits) for num in row]
            for row in self.matrix
        ])

    def __mul__(self, other):
        """
        Matrix * Matrix & Matrix * Vector multiplication
        """
        assert(isinstance(other, Matrix) or isinstance(other, Vector))

        if isinstance(other, Vector):
            # Vector length == Matrix columns:
            assert(len(other) == self.dimensions()[1])

            return(Vector([
                other.dot(Vector(row))
                for row in self.matrix
            ]))

        if isinstance(other, Matrix):
            # Left matrix columns== Right Matrix row:
            assert(self.dimensions()[1] == other.dimensions()[0])

            return Matrix([
                [
                    Vector(row).dot(
                        Vector([other_row[col_i]
                                for other_row in other.matrix])
                    )
                    for col_i in range(other.dimensions()[1])
                ]
                for row in self.matrix
            ])

    def append_right(self, to_append):
        assert(isinstance(to_append, Vector) or isinstance(to_append, Matrix))
        if isinstance(to_append, Vector):
            # Vector length == # rows
            assert(len(to_append) == self.dimensions()[0])
            return Matrix([
                row + [to_append[i]]
                for i, row in enumerate(self.matrix)
            ])

        else:  # If Matrix:
            # Matrices have the same number of rows
            assert(to_append.dimensions()[0] == self.dimensions()[0])
            return Matrix([
                row + to_append[i]
                for i, row in enumerate(self.matrix)
            ])

    def convert_to_echelon_form(self, verbose=False):
        """
        Convert the matrix to echelon form, for example:
        [[1, #, #, #, ...],
         [0, 1, #, #, ...],
         [0, 0, 1, #, ...],
         [0, 0, 0, 1, ...]]
        """
        num_rows, num_columns = self.dimensions()
        assert(num_rows <= num_columns)

        # For convenience
        M = self  # Matrix
        if verbose:
            print("Starting matrix:")
            print(M)

        # For each column in the matrix up to the number of rows...
        for c in range(num_rows):
            # 1. Normalize all rows with non-zero entries in that column to a
            #    value of 1 in the column
            for r in range(c, num_rows):
                if M[r][c] != 0:
                    M[r] = (Vector(M[r]) / M[r][c]).vector

            if verbose:
                print(f"After normalizing column {c}:")
                print(M)

            # 2. If the column's diagonal value == 0, find first row below where
            #    that column's value is 1 and add it to the row with the
            #    diagonal value
            if M[c][c] == 0:
                for r in range(c + 1, num_rows):
                    if M[r][c] == 1:
                        M[c] = (Vector(M[c]) + Vector(M[r])).vector
                        break

                if verbose:
                    print(f"After dealing with a zero diagonal value in column {c}:")
                    print(M)

            if M[c][c] == 0:  # If that columns diagonal value is still 0
                return False # No solution

            # 3. Convert all rows below the row with the diagonal value to have
            #    values of that in column by subtracting the diagonal row from
            #    them
            if (c + 1) < num_rows:  # If not on last column in for loop
                for r in range(c + 1, num_rows):
                    if M[r][c] == 1:
                        M[r] = (Vector(M[r]) - Vector(M[c])).vector

            if verbose:
                print(f"After values in rows below column {c} diagonal to zero:")
                print(M)

        if verbose:
            print(f"Final:")
            print(M)

        return M

    def in_echelon_form(self):
        num_rows, num_columns = self.dimensions()

        if num_rows > num_columns:
            return False
        else:
            for r in range(num_rows):
                if self[r][r] != 1:
                    return False

                if r > 0:
                    for c in range(r):
                        if self[r][c] != 0:
                            return False

        return True

    def reduce_echelon_to_identiy(self, verbose=False):
        """
        Convert the matrix from echelon form:
        [[1, #, #, #, ...],
         [0, 1, #, #, ...],
         [0, 0, 1, #, ...],
         [0, 0, 0, 1, ...]]

        ...to idenity [matrix] form via backward substituion:
        [[1, 0, 0, 0, ...],
         [0, 1, 0, 0, ...],
         [0, 0, 1, 0, ...],
         [0, 0, 0, 1, ...]]
        """
        assert(self.in_echelon_form())

        num_rows, num_columns = self.dimensions()

        # For convenience
        M = self  # Matrix
        if verbose:
            print("Starting matrix:")
            print(M)

        for c in reversed(range(num_rows)):
            # For each column in the orignal matrix, up to the number of rows
            # (in reverse order):
            if c > 0:  # If not the first column
                for r in range(c):
                    if M[r][c] != 0:
                        M[r] = (Vector(M[r]) - M[r][c] * Vector(M[c])).vector

                if verbose:
                    print("After reducing column {c}:")
                    print(M)

        if verbose:
            print("Final:")
            print(M)

        return M

    def invert(self, verbose=False):
        num_rows, num_columns = self.dimensions()
        assert(num_rows == num_columns)  # Is square matrix

        # 1. Append the Identity Matrix of the same dimensions as self
        combined_matrix = self.append_right(IdentityMatrix(num_rows))
        CM = combined_matrix

        # 2. Reduce the combined matrix to "echelon" form
        CM = CM.convert_to_echelon_form(verbose)

        # 3. "Solve" the matrix by backward substituion
        CM = CM.reduce_echelon_to_identiy(verbose)

        # 4. The original Idenity Matrix we appended is now transformed into the
        #    inverse Matrix
        return Matrix([
            row[-num_rows:]
            for row in CM.matrix
        ])

def IdentityMatrix(size):
    """
    Generate the size rows by size columns Identity Matrix, for example when
    size = 3:
    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]
    """
    assert(isinstance(size, int) and size >= 1)

    # Initialize square matrix of all zeroes:
    matrix = Matrix([
        [0] * size
        for _ in range(size)
    ])

    # Set diagonal values to 1:
    for i in range(size):
        matrix[i][i] = 1

    return matrix

def solve_matrix_equation(matrix, vector, verbose=False):
    """
    Solves a linear matrix equation (system of linear equations) of the form
    Ax = b, for example:
     - 1*x + 2*y = 8
     - 8*x + 1*y = 19

    ...or in matrix form:
     [[1, 2] * [x, = [8,
      [8, 1]]   y]    19]

    Requirements:
     - `matrix` must be square and all its columns must be linearly independent
       for the equation to have a solution
     - `vector` must be of the same length as the number of rows in `matrix`
    """
    assert(isinstance(matrix, Matrix))
    assert(isinstance(vector, Vector))
    num_rows, num_columns = matrix.dimensions()
    assert(num_rows == num_columns)  # Is square matrix
    # Check if all `matrix` columns linearly independent?
    assert(len(vector) == num_rows)

    # 1. For ease of computation add `vector` as a column at the end of `matrix`
    combined_matrix = matrix.append_right(vector)
    CM = combined_matrix

    if verbose:
        print(CM)

    # 2. Reduce the combined matrix to "echelon" form
    CM = CM.convert_to_echelon_form(verbose)

    if verbose:
        print(CM)

    # 3. Solve the matrix by backward substituion
    CM = CM.reduce_echelon_to_identiy(verbose)

    if verbose:
        print(CM)

    # 4. The values in the final, appended column are the solution
    return Vector([row[num_columns] for row in CM.matrix])
