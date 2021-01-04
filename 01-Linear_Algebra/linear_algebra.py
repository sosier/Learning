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
