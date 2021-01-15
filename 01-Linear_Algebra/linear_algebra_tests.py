from math import sqrt, pi, cos
import traceback

from linear_algebra import is_numeric, Vector, Matrix, solve_matrix_equation

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

        assert(Vector(1, 0).orthogonal_to(Vector(0, 1)) is True)
        assert(Vector(1, 0).orthogonal_to(Vector(1, 1)) is False)
        assert(Vector(1, 0).orthogonal_to(Vector(1, 0)) is False)

        # Projections:
        assert(Vector(2, 0).normalize() == Vector(1, 0))
        assert(Vector(2227.9, 0).normalize() == Vector(1, 0))
        assert(Vector(0, 2).normalize() == Vector(0, 1))
        assert(Vector(10, 5, -6).scalar_projection_onto(Vector(3, -4, 0)) == 2)
        assert(
            round(
                Vector(10, 5, -6).vector_projection_onto(Vector(3, -4, 0)),
                PRECISION
            )
            == Vector(6/5, -8/5, 0)
        )
        assert(
            Vector(10, 5, -6).vector_projection_scalar(Vector(3, -4, 0)) == 2/5
        )

        # Changing basis:
        assert(
            Vector(3, 4).change_basis(Vector(2, 1), Vector(-2, 4))
            == Vector(2, 1/2)
        )
        assert(
            round(
                Vector(5, -1).change_basis(Vector(1, 1), Vector(1, -1)),
                PRECISION
            )
            == Vector(2, 3)
        )
        assert(
            Vector(10, -5).change_basis(Vector(3, 4), Vector(4, -3))
            == Vector(2/5, 11/5)
        )
        assert(
            round(
                Vector(2, 2).change_basis(Vector(-3, 1), Vector(1, 3)),
                PRECISION
            )
            == Vector(-2/5, 4/5)
        )
        assert(
            round(
                Vector(1, 1, 1).change_basis(
                    Vector(2, 1, 0), Vector(1, -2, -1), Vector(-1, 2, -5)
                ),
                PRECISION
            )
            == round(Vector(3/5, -1/3, -2/15), PRECISION)
        )
        assert(
            Vector(1, 1, 2, 3).change_basis(
                Vector(1, 0, 0, 0),
                Vector(0, 2, -1, 0),
                Vector(0, 1, 2, 0),
                Vector(0, 0, 0, 3)
            )
            == Vector(1, 0, 1, 1)
        )

        print("All Vector tests pass")
        return True
    except Exception:
        print("Vector TEST FAILED!")
        traceback.print_exc()
        return False

def test_Matrix():
    try:
        # Check allowable input types:
        m = Matrix(
            [1, 2, 3],
            [1, 2, 3],
            [1, 2, 3]
        )
        m = Matrix([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])

        assert(m.dimensions() == (3, 3))
        assert(Matrix([
                [1, 2, 3],
                [1, 2, 3]
            ]).dimensions() == (2, 3)
        )

        # Check equalities:
        assert((m == m) is True)
        assert((m != m) is False)
        assert((m == 1) is False)
        assert((m != 1) is True)
        assert(Matrix([1, 2, 3]) != Matrix([3, 2, 1]))
        assert(Matrix([1, 2, 3]) != Vector([1, 2, 3]))
        assert(
            Matrix([1, 2, 3])
            != Matrix([1], [2], [3])
        )

        # Check number getting / setting:
        assert(m[2][0] == 7)
        m[2][0] = 0
        assert(m == Matrix([
            [1, 2, 3],
            [4, 5, 6],
            [0, 8, 9]
        ]))

        # Rounding
        assert(
            round(Matrix([
                [1.1, 2.9],
                [4.7, 5.3]
            ]), 0)
            == Matrix([
                [1, 3],
                [5, 5]
            ])
        )

        # Multiplication:
        assert(
            Matrix([
                [1, 2],
                [3, 4]
            ]) * Vector(1, 2)
            == Vector(5, 11)
        )
        assert(
            Matrix([
                [1, 2],
                [3, 4]
            ]) * Matrix([
                [1, 2],
                [3, 4]
            ])
            == Matrix([
                [7,  10],
                [15, 22]
            ])
        )

        # Solving:
        assert(
            Matrix(
                [1, 2],
                [8, 1]
            ).convert_to_echelon_form()
            == Matrix(
                [1, 2],
                [0, 1]
            )
        )

        print("All Matrix tests pass")
        return True
    except Exception:
        print("Matrix TEST FAILED!")
        traceback.print_exc()
        return False

def test_solve_matrix_equation():
    try:
        assert(
            solve_matrix_equation(
                Matrix(
                    [1, 2],
                    [8, 1]
                ),
                Vector(8, 19)
            )
            == Vector(2, 3)
        )
        assert(
            round(
                solve_matrix_equation(
                    Matrix(
                        [4, 6, 2],
                        [3, 4, 1],
                        [2, 8, 13]
                    ),
                    Vector(9, 7, 2)
                ),
                PRECISION
            )
            == Vector(3, -1/2, 0)
        )
        assert(
            solve_matrix_equation(
                Matrix(
                    [1, 1, 1],
                    [3, 2, 1],
                    [2, 1, 2]
                ),
                Vector(15, 28, 23)
            )
            == Vector(3, 7, 5)
        )

        print("All solve_matrix_equation tests pass")
        return True
    except Exception:
        print("solve_matrix_equation TEST FAILED!")
        traceback.print_exc()
        return False


test_is_numeric()
test_Vector()
test_Matrix()
test_solve_matrix_equation()
