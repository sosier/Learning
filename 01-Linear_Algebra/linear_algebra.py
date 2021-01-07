from math import sqrt, acos, degrees

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