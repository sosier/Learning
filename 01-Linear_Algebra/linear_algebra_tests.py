from math import sqrt, pi, cos
import traceback

from linear_algebra import is_numeric, Vector

PRECISION = 10

def test_is_numeric():
    try:
        assert(is_numeric(1) is True)
        assert(is_numeric(1.0) is True)
        assert(is_numeric(1.01) is True)
        assert(is_numeric("1") is False)
        assert(is_numeric([1]) is False)

        print("All is_numeric tests pass")
        return True
    except Exception:
        print("is_numeric TEST FAILED!")
        traceback.print_exc()
        return False

def test_Vector():
    try:
        # Check allowable input types:
        v = Vector(1, 2, 3)
        v = Vector([1, 2, 3])

        assert(len(v) == 3)

        # Check equalities:
        assert((v == v) is True)
        assert((v != v) is False)
        assert((v == 1) is False)
        assert((v != 1) is True)
        assert(Vector(1, 2, 3) != Vector(3, 2, 1))

        # Check number getting / setting:
        assert(v[2] == 3)
        v[2] = 2
        assert(v == Vector(1, 2, 2))

        # Check vector math:
        assert((v + v) == Vector(2, 4, 4))
        assert((v - v) == Vector(0, 0, 0))
        assert((3 * v) == Vector(3, 6, 6))
        assert((v * 3) == Vector(3, 6, 6))
        assert((v / 3) == Vector(1/3, 2/3, 2/3))

        # Vector magnitudes & dot products:
        assert(Vector(3, 4).magnitude() == 5)
        assert(Vector(3, 0).magnitude() == 3)
        assert(Vector(0, 4).magnitude() == 4)
        assert(v.dot_product(v) == 9)
        assert(Vector(1, 2).dot_product(Vector(3, 4)) == 11)
        assert(v.dot(v) == 9)
        assert(Vector(1, 2).dot(Vector(3, 4)) == 11)
        assert(sqrt(v.dot_product(v)) == v.magnitude())
        assert(sqrt(v.dot(v)) == v.magnitude())

        # Angle / cosines similarity metrics:
        assert(Vector(1, 0).cosine_similarity(Vector(1, 0)) == 1)
        assert(Vector(1, 0).cosine_distance(Vector(1, 0)) == 0)
        assert(Vector(1, 0).angle_between(Vector(1, 0)) == 0)
        assert(Vector(1, 0).angle_between(Vector(1, 0), radians=True) == 0)
        assert(Vector(1, 0).cosine_similarity(Vector(0, 1)) == 0)
        assert(Vector(1, 0).cosine_distance(Vector(0, 1)) == 1)
        assert(Vector(1, 0).angle_between(Vector(0, 1)) == 90)
        assert(Vector(1, 0).angle_between(Vector(0, 1), radians=True) == pi/2)
        assert(Vector(1, 0).cosine_similarity(Vector(-1, 0)) == -1)
        assert(Vector(1, 0).cosine_distance(Vector(-1, 0)) == 2)
        assert(Vector(1, 0).angle_between(Vector(-1, 0)) == 180)
        assert(Vector(1, 0).angle_between(Vector(-1, 0), radians=True) == pi)
        assert(
            round(Vector(1, 0).cosine_similarity(Vector(1, 1)), PRECISION)
            == round(cos(pi/4), PRECISION)
        )
        assert(
            round(Vector(1, 0).cosine_distance(Vector(1, 1)), PRECISION)
            == round(1 - cos(pi/4), PRECISION)
        )
        assert(int(Vector(1, 0).angle_between(Vector(1, 1))) == 45)
        assert(
            round(
                Vector(1, 0).angle_between(Vector(1, 1), radians=True),
                PRECISION
            )
            == round(pi/4, PRECISION)
        )

        print("All Vector tests pass")
        return True
    except Exception:
        print("Vector TEST FAILED!")
        traceback.print_exc()
        return False

test_is_numeric()
test_Vector()
