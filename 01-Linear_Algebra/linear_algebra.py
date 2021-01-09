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

    def __round__(self, ndigits):
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
