def multiply_range(from_:int, to:int):
    product = 1

    for i in range(from_, to + 1):
        product = product * i

    return product

def factorial(n:int):
    assert(n >= 0)
    if n == 0:
        return 1
    else:
        return multiply_range(1, n)

def naive_probability(num_desired_outcomes:int, num_possible_outcomes:int):
    """
    Assumes:
     1. All outcomes equally likely
     2. Finite sample (outcome) space
    """
    assert(num_desired_outcomes >= 0)
    assert(num_possible_outcomes >= 0)
    assert(num_desired_outcomes <= num_possible_outcomes)

    return num_desired_outcomes / num_possible_outcomes

def n_choose_k(n:int, k:int):
    assert(n >= 0)
    assert(k >= 0)
    if k > n:
        return 0

    return (
        factorial(n)
        / (factorial(n - k) * factorial(k))
    )

def num_sample_possibilities(
        n:int, k:int, with_replacement=False, order_matters=False
    ):
    assert(n >= 0)
    assert(k >= 0)

    if with_replacement and order_matters:
        return n**k
    elif with_replacement and not order_matters:
        return n_choose_k(n + k - 1, k)
    elif not with_replacement and order_matters:
        if k > n:
            return 0
        return factorial(n) / factorial(n - k)
    elif not with_replacement and not order_matters:
        return n_choose_k(n, k)
