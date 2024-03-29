from math import sqrt, acos, degrees
from itertools import combinations


def is_numeric(object):
    return type(object) == int or type(object) == float


class Vector:
    """
    A vector is an ordered, list of numbers
    """

    def __init__(self, *numbers):
        if len(numbers) == 1 and type(numbers[0]) == list:
            numbers = numbers[0]

        assert all(is_numeric(num) for num in numbers)
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
                # fmt: off
                len(self) == len(other)
                and all(x == y for x, y in zip(self, other))
                # fmt: on
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
        assert isinstance(other, Vector)
        assert len(self) == len(other)
        return Vector(*[x + y for x, y in zip(self, other)])

    def __sub__(self, other):
        assert isinstance(other, Vector)
        assert len(self) == len(other)
        return Vector(*[x - y for x, y in zip(self, other)])

    def __mul__(self, other):
        assert is_numeric(other) or isinstance(other, Vector)
        if is_numeric(other):
            return Vector(*[other * num for num in self.vector])
        else:
            # fmt: off
            return Vector(*[
                x * y
                for x, y in zip(self.vector, other.vector)
            ])
            # fmt: on

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        assert is_numeric(other)
        # fmt: off
        return Vector(*[1/other * num for num in self.vector])
        # fmt: on

    def __round__(self, ndigits=None):
        return Vector(*[round(num, ndigits) for num in self.vector])

    def magnitude(self):
        """
        "Magnitude", "size", or "length" of the vector.

        Calculated via the Pythagorean thereom. In that way, this is also the
        distance from the origin to the end of the vector.
        """
        # fmt: off
        return sqrt(sum([num**2 for num in self.vector]))
        # fmt: on

    def dot_product(self, other):
        """
        "Dot product" or "sum product" of two vectors
        """
        assert isinstance(other, Vector)
        assert len(self) == len(other)
        return sum([x * y for x, y in zip(self, other)])

    def dot(self, other):
        """
        Short convenience function for accessing `.dot_product(...)` since this
        is such a common vector operation
        """
        return self.dot_product(other)

    def cosine_similarity(self, other):
        # fmt: off
        return (
            self.dot(other) /
            (self.magnitude() * other.magnitude())
        )
        # fmt: on

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
        return self.dot(other) / other.magnitude()

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
        assert all([isinstance(vector, Vector) for vector in basis_vectors])
        assert all([len(self) == len(vector) for vector in basis_vectors])
        # fmt: off
        assert all([
            # All basis vectors must be orthogonal to eachother
            vector_a.orthogonal_to(vector_b)
            for vector_a, vector_b in combinations(basis_vectors, 2)
        ])
        return Vector(*[
            self.vector_projection_scalar(basis_vector)
            for basis_vector in basis_vectors
        ])
        # fmt: on


class Matrix:
    """
    A matrix is an ordered, "rectangular" list of numbers (with rows and columns)
    """

    def __init__(self, *rows_of_numbers):
        if (
            len(rows_of_numbers) == 1
            and type(rows_of_numbers[0]) == list
            and (
                len(rows_of_numbers[0]) == 0
                or type(rows_of_numbers[0][0]) == list
            )
        ):
            rows_of_numbers = rows_of_numbers[0]

        for row in rows_of_numbers:
            assert type(row) == list
            assert all(is_numeric(num) for num in row)
            assert len(row) == len(rows_of_numbers[0])

        self.matrix = list(rows_of_numbers)

    def __str__(self):
        return "\n".join([str(row) for row in self.matrix]) + "\n"

    @property
    def num_rows(self):
        return len(self.matrix)

    @property
    def num_columns(self):
        return len(self.matrix[0])

    def dimensions(self):
        """
        Return the matrix dimensions (m x n) / (rows x columns) as a tuple:
        (m, n) / (rows, columns)
        """
        return (self.num_rows, self.num_columns)

    def is_square(self):
        return self.num_rows == self.num_columns

    def transpose(self):
        # fmt: off
        return Matrix([
            [self[r][c] for r in range(self.num_rows)]
            for c in range(self.num_columns)
        ])
        # fmt: on

    @property
    def T(self):
        """
        Alias for Matrix.transpose()
        """
        return self.transpose()

    def is_orthonormal(self):
        column_vectors = self.T.matrix
        # fmt: off
        return (
            all([
                # All column vectors must be orthogonal to eachother
                Vector(vector_a).orthogonal_to(Vector(vector_b))
                for vector_a, vector_b in combinations(column_vectors, 2)
            ])
            and all([
                # All column vectors must be normalized: length (magnitude) = 1
                Vector(vector).magnitude() == 1
                for vector in column_vectors
            ])
        )
        # fmt: on

    def __eq__(self, other):
        try:
            # fmt: off
            return (
                self.dimensions() == other.dimensions()
                and all(
                    self_val == other_val
                    for self_row, other_row in zip(self.matrix, other.matrix)
                    for self_val, other_val in zip(self_row, other_row)
                )
            )
            # fmt: on
        except:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getitem__(self, i):
        return self.matrix[i]

    def __setitem__(self, i, value):
        self.matrix[i] = value

    def __round__(self, ndigits=None):
        # fmt: off
        return Matrix([
            [round(num, ndigits) for num in row]
            for row in self.matrix
        ])
        # fmt: on

    def __add__(self, other):
        assert isinstance(other, Matrix)
        # fmt: off
        return Matrix([
            [
                val_self + val_other
                for val_self, val_other in zip(row_self, row_other)
            ]
            for row_self, row_other in zip(self.matrix, other.matrix)
        ])
        # fmt: on

    def __mul__(self, other):
        """
        Matrix * Matrix & Matrix * Vector multiplication
        """
        assert isinstance(other, Matrix) or isinstance(other, Vector)

        if isinstance(other, Vector):
            # Vector length == Matrix columns:
            assert len(other) == self.dimensions()[1]
            # fmt: off
            return Vector([
                other.dot(Vector(row))
                for row in self.matrix
            ])
            # fmt: on

        if isinstance(other, Matrix):
            # Left matrix columns== Right Matrix row:
            assert self.dimensions()[1] == other.dimensions()[0]
            # fmt: off
            return Matrix([
                [
                    Vector(row).dot(
                        Vector([
                            other_row[col_i]
                            for other_row in other.matrix
                        ])
                    )
                    for col_i in range(other.dimensions()[1])
                ]
                for row in self.matrix
            ])
            # fmt: on

    def __rmul__(self, other):
        assert is_numeric(other)
        # fmt: off
        return Matrix([
            [other * val for val in row]
            for row in self.matrix
        ])
        # fmt: on

    def append_right(self, to_append):
        assert isinstance(to_append, Vector) or isinstance(to_append, Matrix)
        if isinstance(to_append, Vector):
            # Vector length == # rows
            assert len(to_append) == self.dimensions()[0]
            # fmt: off
            return Matrix([
                row + [to_append[i]]
                for i, row in enumerate(self.matrix)
            ])
            # fmt: on

        else:  # If Matrix:
            # Matrices have the same number of rows
            assert to_append.dimensions()[0] == self.dimensions()[0]
            # fmt: off
            return Matrix([
                row + to_append[i]
                for i, row in enumerate(self.matrix)
            ])
            # fmt: on

    def convert_to_echelon_form(self, verbose=False):
        """
        Convert the matrix to echelon form, for example:
        [[1, #, #, #, ...],
         [0, 1, #, #, ...],
         [0, 0, 1, #, ...],
         [0, 0, 0, 1, ...]]
        """
        num_rows, num_columns = self.dimensions()
        assert num_rows <= num_columns

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
                    print(
                        f"After dealing with a zero diagonal value in column {c}:"
                    )
                    print(M)

            if M[c][c] == 0:  # If that columns diagonal value is still 0
                return False  # No solution

            # 3. Convert all rows below the row with the diagonal value to have
            #    values of that in column by subtracting the diagonal row from
            #    them
            if (c + 1) < num_rows:  # If not on last column in for loop
                for r in range(c + 1, num_rows):
                    if M[r][c] == 1:
                        M[r] = (Vector(M[r]) - Vector(M[c])).vector

            if verbose:
                print(
                    f"After values in rows below column {c} diagonal to zero:"
                )
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
        assert self.in_echelon_form()

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
        assert num_rows == num_columns  # Is square matrix

        # 1. Append the Identity Matrix of the same dimensions as self
        combined_matrix = self.append_right(IdentityMatrix(num_rows))
        CM = combined_matrix

        # 2. Reduce the combined matrix to "echelon" form
        CM = CM.convert_to_echelon_form(verbose)

        # 3. "Solve" the matrix by backward substituion
        CM = CM.reduce_echelon_to_identiy(verbose)

        # 4. The original Idenity Matrix we appended is now transformed into the
        #    inverse Matrix
        # fmt: off
        return Matrix([
            row[-num_rows:]
            for row in CM.matrix
        ])
        # fmt: on

    def determinant(self):
        assert self.is_square()
        assert self.num_rows > 0

        if self.num_rows == 1:
            return self[0][0]
        else:
            # Recurse to get final determinant:
            # fmt: off
            return sum([
                (-1 if c % 2 == 1 else 1)  # 1 if even, -1 if odd
                * self[0][c]
                # x Determinant of the submatrix left if you remove the top row
                # and column `c`:
                * Matrix([
                    [
                        self[r][col]
                        for col in range(self.num_columns)
                        if c != col
                    ]
                    for r in range(1, self.num_rows)  # Removes top row
                ]).determinant()
                for c in range(self.num_columns)
            ])
            # fmt: on

    def to_orthonormal(self):
        """
        Convert the current matrix to be orthonormal using the Gram–Schmidt
        process
        """
        if self.is_orthonormal():
            return self
        else:
            # Column vectors must be linearly independent:
            assert self.determinant() != 0

            column_vectors = [Vector(vector) for vector in self.T.matrix]

            for c, vector in enumerate(column_vectors):
                if c == 0:
                    # 1. Normalize the first vector to length 1 (unit length).
                    #    This will be the first orthonormal basis vector
                    column_vectors[c] = vector.normalize()
                else:
                    # 2. For all subsequent vectors, first reduce that vector to
                    #    only the orthogonal portion of its direction by
                    #    subtracting it's vector projections onto each of the
                    #    orthonormal bases found so far. The remainder after all
                    #    subtraction are complete will be only the component of
                    #    that vector orthogonal to all previously calculated
                    #    basis vectors:
                    for basis_vector in column_vectors[:c]:
                        vector = vector - vector.vector_projection_onto(
                            basis_vector
                        )

                    # 3. Finally, normalize the now orthogonal vector to be unit
                    #    length:
                    column_vectors[c] = vector.normalize()

            # Convert the Vector objects back to lists:
            column_vectors = [vector.vector for vector in column_vectors]

            # Return the transpose since Matrix accept row vectors as input,
            # not column vectors:
            return Matrix(column_vectors).T

    # Will NOT be implementing code to find eigenvalues / eigenvectors. This is
    # very complicated to do for a Matrix of arbitrary size. For example:
    # http://www.cs.unc.edu/techreports/96-043.pdf
    # def get_eigenvalues_vectors(self):
    #     pass


def IdentityMatrix(size):
    """
    Generate the size rows by size columns Identity Matrix, for example when
    size = 3:
    [[1, 0, 0],
     [0, 1, 0],
     [0, 0, 1]]
    """
    assert isinstance(size, int) and size >= 1

    # Initialize square matrix of all zeroes:
    # fmt: off
    matrix = Matrix([
        [0] * size
        for _ in range(size)
    ])
    # fmt: on

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
    assert isinstance(matrix, Matrix)
    assert isinstance(vector, Vector)
    num_rows, num_columns = matrix.dimensions()
    assert num_rows == num_columns  # Is square matrix
    # Check if all `matrix` columns linearly independent?
    assert len(vector) == num_rows

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


def page_rank(link_matrix, d=0.85, max_iterations=None):
    """
    link_matrix = Square (n by n) matrix of the transition probabilities from
        one site to another
    d = Dampening factor (probability of continuing to "click links" vs.
        randomly going to a site). Used to make sure algorithm doesn't get
        stuck, for example, on a site with no outbound links
    max_iterations = Maximum number of algorithm iterations allowed before a
        result is returned. If an exact solution is found is less iterations,
        that solution will be returned and no more iterations will be completed.
    """
    assert isinstance(link_matrix, Matrix)
    assert link_matrix.is_square()
    # fmt: off
    assert all(
        sum(column) == 1
        for column in link_matrix.T.matrix
    )
    # fmt: on
    assert 0 <= d <= 1
    assert max_iterations is None or max_iterations >= 0

    n = link_matrix.num_rows
    i = 0
    # Build link_matrix with dampening (`d`)
    # fmt: off
    LM = (
        d * link_matrix
        + (
            ((1 - d) / n)
            * Matrix([[1] * n] * n)   # * 1's matrix
        )
    )
    # fmt: on
    # Initalize all sites to the same page rank:
    rank_values = Vector([1] * link_matrix.num_rows) / link_matrix.num_rows
    last_rank_values = None

    if max_iterations is None or max_iterations > 0:
        while rank_values != last_rank_values:
            i += 1
            last_rank_values = rank_values
            rank_values = LM * rank_values

            if max_iterations is not None and i >= max_iterations:
                return rank_values

    return rank_values
