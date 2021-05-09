"""
Test probability.py

To test run `pytest` in the command line
"""
from probability import (
    multiply_range, factorial, naive_probability, n_choose_k,
    num_sample_possibilities
)

def test_multiply_range():
    assert(multiply_range(1, 3) == 6)
    assert(multiply_range(1, 5) == 120)
    assert(multiply_range(3, 4) == 12)

def test_factorial():
    assert(factorial(3) == 6)
    assert(factorial(5) == 120)
    assert(factorial(0) == 1)
    assert(factorial(1) == 1)

def test_naive_probability():
    assert(naive_probability(1, 2) == 0.5)
    assert(naive_probability(0, 2) == 0)
    assert(naive_probability(6, 8) == 0.75)

def test_n_choose_k():
    assert(n_choose_k(12, 0) == 1)
    assert(n_choose_k(12, 12) == 1)
    assert(n_choose_k(12, 13) == 0)
    assert(n_choose_k(6, 3) == 20)

def test_num_sample_possibilities():
    # Standard
    assert(
        num_sample_possibilities(
            n=10, k=3, with_replacement=True, order_matters=True
        ) == 1000
    )
    assert(
        num_sample_possibilities(
            n=3, k=3, with_replacement=True, order_matters=False
        ) == 10
    )
    assert(
        num_sample_possibilities(
            n=10, k=2, with_replacement=False, order_matters=True
        ) == 90
    )
    assert(
        num_sample_possibilities(
            n=7, k=3, with_replacement=False, order_matters=False
        ) == 35
    )
    # k > n
    assert(
        num_sample_possibilities(
            n=2, k=3, with_replacement=True, order_matters=True
        ) == 8
    )
    assert(
        num_sample_possibilities(
            n=2, k=3, with_replacement=True, order_matters=False
        ) == 4
    )
    assert(
        num_sample_possibilities(
            n=2, k=3, with_replacement=False, order_matters=True
        ) == 0
    )
    assert(
        num_sample_possibilities(
            n=2, k=3, with_replacement=False, order_matters=False
        ) == 0
    )
    # k = 0
    assert(
        num_sample_possibilities(
            n=2, k=0, with_replacement=True, order_matters=True
        ) == 1
    )
    assert(
        num_sample_possibilities(
            n=2, k=0, with_replacement=True, order_matters=False
        ) == 1
    )
    assert(
        num_sample_possibilities(
            n=2, k=0, with_replacement=False, order_matters=True
        ) == 1
    )
    assert(
        num_sample_possibilities(
            n=2, k=0, with_replacement=False, order_matters=False
        ) == 1
    )
