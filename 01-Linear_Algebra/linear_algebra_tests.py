import traceback

from linear_algebra import is_numeric, Vector

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

        print("All Vector tests pass")
        return True
    except Exception:
        print("Vector TEST FAILED!")
        traceback.print_exc()
        return False

test_is_numeric()
test_Vector()
