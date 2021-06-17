"""
Test probability.py

To test run `pytest` in the command line
"""
import numpy as np

from probability import (
    multiply_range, factorial, naive_probability, n_choose_k,
    num_sample_possibilities, simulate_birthday_problem,
    probability_two_aces_info_comparison,
    simulate_probability_has_disease_given_positive_medical_test,
    simulate_monty_hall_problem, Bernoulli, Binomial, Hypergeometric, Geometric,
    NegativeBinomial
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

def test_simulate_birthday_problem():
    np.random.seed(12345)
    # Assert estimated probability is within 1% of 50%
    assert(abs(simulate_birthday_problem(23) - 0.50) < 0.01)

    # Assert estimated probability is within 0.1% of 99.9%:
    assert(abs(simulate_birthday_problem(70) - 0.999) < 0.001)

def test_probability_two_aces_info_comparison():
    assert(
        probability_two_aces_info_comparison()
        == {
            "prob_two_aces_given_one_ace": 1/33,
            "prob_two_aces_given_ace_of_spades": 1/17
        }
    )

def test_simulate_probability_has_disease_given_positive_medical_test():
    np.random.seed(12345)
    # Assert estimated probability is within 0.5% of 16%
    assert(
        abs(
            simulate_probability_has_disease_given_positive_medical_test(
                num_simulations=100000
            ) - 0.16
        ) < 0.005
    )

def test_simulate_monty_hall_problem():
    np.random.seed(12345)
    # Assert estimated probability is within 1% of 66.666...%
    assert(abs(simulate_monty_hall_problem(num_simulations=10000) - 2/3) < 0.02)

def test_Bernoulli():
    # Test .prob_of()
    assert(Bernoulli(p=1).prob_of(1) == 1)
    assert(Bernoulli(p=1).prob_of(0) == 0)
    assert(Bernoulli(p=0.7).prob_of(1) == 0.7)
    assert(round(Bernoulli(p=0.7).prob_of(0), 10) == 0.3)

    # Test .sample()
    assert(Bernoulli(p=1).sample() == 1)
    assert(all(Bernoulli(p=1).sample(10) == 1))

    np.random.seed(12345)
    assert(Bernoulli(p=0.7).sample() == 0)
    assert(np.array_equal(
        Bernoulli(p=0.7).sample(3),
        np.array([1, 1, 1])
    ))

    # Test .expected_value()
    assert(Bernoulli(p=1).expected_value() == 1)
    assert(Bernoulli(p=0).expected_value() == 0)
    assert(Bernoulli(p=0.7).expected_value() == 0.7)

def test_Binomial():
    # Test .prob_of()
    assert(Binomial(n=1, p=1).prob_of(1) == 1)
    assert(Binomial(n=1, p=1).prob_of(0) == 0)
    assert(Binomial(n=1, p=0.7).prob_of(1) == 0.7)
    assert(round(Binomial(n=1, p=0.7).prob_of(0), 10) == 0.3)
    assert(Binomial(n=5, p=1).prob_of(0) == 0)
    assert(Binomial(n=5, p=1).prob_of(1) == 0)
    assert(Binomial(n=5, p=1).prob_of(2) == 0)
    assert(Binomial(n=5, p=1).prob_of(3) == 0)
    assert(Binomial(n=5, p=1).prob_of(4) == 0)
    assert(Binomial(n=5, p=1).prob_of(5) == 1)
    assert(Binomial(n=5, p=1).prob_of(6) == 0)
    assert(Binomial(n=3, p=0.5).prob_of(0) == 1/8)
    assert(Binomial(n=3, p=0.5).prob_of(1) == 3/8)
    assert(Binomial(n=3, p=0.5).prob_of(2) == 3/8)
    assert(Binomial(n=3, p=0.5).prob_of(3) == 1/8)
    assert(Binomial(n=3, p=0.5).prob_of(4) == 0)

    # Test .sample()
    assert(Binomial(n=1, p=1).sample() == 1)
    assert(all(Binomial(n=1, p=1).sample(10) == 1))
    assert(Binomial(n=5, p=1).sample() == 5)
    assert(all(Binomial(n=5, p=1).sample(10) == 5))

    np.random.seed(12345)
    assert(Binomial(n=1, p=0.7).sample() == 0)
    assert(np.array_equal(
        Binomial(n=1, p=0.7).sample(3),
        np.array([1, 1, 1])
    ))
    assert(Binomial(n=3, p=0.5).sample() == 0)
    assert(np.array_equal(
        Binomial(n=3, p=0.5).sample(3),
        np.array([0, 1, 2])
    ))

    # Test .expected_value()
    assert(Binomial(n=1, p=1).expected_value() == 1)
    assert(Binomial(n=8, p=1).expected_value() == 8)
    assert(Binomial(n=1, p=0).expected_value() == 0)
    assert(Binomial(n=8, p=0).expected_value() == 0)
    assert(Binomial(n=8, p=0.5).expected_value() == 4)

def test_Hypergeometric():
    # Test .prob_of()
    assert(Hypergeometric(N=0, K=0, n=0).prob_of(0) == 1)
    assert(Hypergeometric(N=0, K=0, n=0).prob_of(1) == 0)
    assert(Hypergeometric(N=10, K=0, n=8).prob_of(0) == 1)
    assert(Hypergeometric(N=10, K=0, n=8).prob_of(1) == 0)
    assert(Hypergeometric(N=10, K=10, n=8).prob_of(8) == 1)
    assert(Hypergeometric(N=10, K=10, n=8).prob_of(7) == 0)
    assert(Hypergeometric(N=10, K=10, n=8).prob_of(9) == 0)
    assert(Hypergeometric(N=3, K=2, n=3).prob_of(2) == 1)
    assert(Hypergeometric(N=3, K=2, n=3).prob_of(1) == 0)
    assert(Hypergeometric(N=3, K=2, n=3).prob_of(3) == 0)
    assert(Hypergeometric(N=3, K=2, n=2).prob_of(0) == 0)
    assert(Hypergeometric(N=3, K=2, n=2).prob_of(1) == 2/3)
    assert(Hypergeometric(N=3, K=2, n=2).prob_of(2) == 1/3)
    assert(Hypergeometric(N=3, K=2, n=2).prob_of(3) == 0)

    # Test .sample()
    assert(Hypergeometric(N=1, K=1, n=1).sample() == 1)
    assert(Hypergeometric(N=1, K=0, n=1).sample() == 0)
    assert(np.array_equal(
        Hypergeometric(N=1, K=1, n=1).sample(3),
        np.ones(3)
    ))
    assert(np.array_equal(
        Hypergeometric(N=1, K=0, n=1).sample(3),
        np.zeros(3)
    ))
    assert(Hypergeometric(N=5, K=5, n=3).sample() == 3)
    assert(Hypergeometric(N=5, K=0, n=3).sample() == 0)
    assert(all(Hypergeometric(N=5, K=5, n=3).sample(3) == 3))
    assert(all(Hypergeometric(N=5, K=0, n=3).sample(3) == 0))

    np.random.seed(12345)
    assert(Hypergeometric(N=3, K=2, n=2).sample() == 2)
    assert(np.array_equal(
        Hypergeometric(N=3, K=2, n=2).sample(3),
        np.array([1, 1, 2])
    ))

    # Test .expected_value()
    assert(Hypergeometric(N=3, K=2, n=2).expected_value() == 4/3)
    assert(Hypergeometric(N=3, K=0, n=2).expected_value() == 0)
    assert(Hypergeometric(N=3, K=3, n=2).expected_value() == 2)
    assert(Hypergeometric(N=3, K=3, n=1).expected_value() == 1)

def test_Geometric():
    # Test .prob_of()
    assert(Geometric(p=0).prob_of(12) == 0)
    assert(Geometric(p=0).prob_of(0) == 0)
    assert(Geometric(p=1).prob_of(0) == 1)
    assert(Geometric(p=1).prob_of(12) == 0)
    assert(Geometric(p=0.5).prob_of(0) == 0.5)
    assert(Geometric(p=0.5).prob_of(1) == 0.25)
    assert(Geometric(p=0.5).prob_of(2) == 0.125)
    assert(Geometric(p=0.2).prob_of(0) == 0.2)
    assert(abs(Geometric(p=0.2).prob_of(1) - 4/25) < 0.00000001)
    assert(abs(Geometric(p=0.2).prob_of(2) - 16/125) < 0.00000001)

    # Test .sample()
    assert(Geometric(p=0).sample() == np.inf)
    assert(all(Geometric(p=0).sample(3) == np.inf))
    assert(Geometric(p=1).sample() == 0)
    assert(all(Geometric(p=1).sample(3) == 0))

    np.random.seed(12345)
    assert(Geometric(p=0.5).sample() == 1)
    assert(np.array_equal(
        Geometric(p=0.5).sample(3),
        np.array([0, 0, 8])
    ))

    assert(Geometric(p=0.8).sample() == 0)
    assert(np.array_equal(
        Geometric(p=0.8).sample(3),
        np.array([0, 0, 3])
    ))

    # Test .expected_value()
    assert(Geometric(p=0.5).expected_value() == 1)
    assert(Geometric(p=0).expected_value() == np.inf)
    assert(Geometric(p=1).expected_value() == 0)
    assert(abs(Geometric(p=1/3).expected_value() - 2) < 0.000000001)

def test_NegativeBinomial():
    # Test .prob_of()
    assert(NegativeBinomial(r=0, p=0).prob_of(1) == 0)
    assert(NegativeBinomial(r=0, p=0).prob_of(12) == 0)
    assert(NegativeBinomial(r=0, p=0).prob_of(0) == 1)
    assert(NegativeBinomial(r=1, p=0).prob_of(0) == 0)
    assert(NegativeBinomial(r=1, p=0).prob_of(1) == 0)
    assert(NegativeBinomial(r=1, p=0).prob_of(2) == 0)
    assert(NegativeBinomial(r=1, p=0).prob_of(np.inf) == 1)
    assert(NegativeBinomial(r=1, p=1).prob_of(0) == 1)
    assert(NegativeBinomial(r=1, p=1).prob_of(1) == 0)
    assert(NegativeBinomial(r=2, p=1).prob_of(0) == 1)
    assert(NegativeBinomial(r=2, p=1).prob_of(1) == 0)
    assert(NegativeBinomial(r=1, p=0.5).prob_of(0) == 0.5)
    assert(NegativeBinomial(r=1, p=0.5).prob_of(1) == 0.25)
    assert(NegativeBinomial(r=1, p=0.5).prob_of(2) == 0.125)
    assert(NegativeBinomial(r=2, p=0.5).prob_of(0) == 0.25)
    assert(NegativeBinomial(r=2, p=0.5).prob_of(1) == 0.25)
    assert(NegativeBinomial(r=2, p=0.5).prob_of(2) == 0.0625 * 3)

    # Test .sample()
    assert(NegativeBinomial(r=0, p=0).sample() == 0)
    assert(all(NegativeBinomial(r=0, p=0).sample(3) == 0))
    assert(NegativeBinomial(r=1, p=0).sample() == np.inf)
    assert(all(NegativeBinomial(r=1, p=0).sample(3) == np.inf))
    assert(NegativeBinomial(r=1, p=1).sample() == 0)
    assert(all(NegativeBinomial(r=1, p=1).sample(3) == 0))
    assert(NegativeBinomial(r=2, p=1).sample() == 0)
    assert(all(NegativeBinomial(r=2, p=1).sample(3) == 0))

    np.random.seed(12345)
    assert(NegativeBinomial(r=1, p=0.5).sample() == 1)
    assert(np.array_equal(
        NegativeBinomial(r=1, p=0.5).sample(3),
        np.array([0, 0, 8])
    ))
    assert(NegativeBinomial(r=2, p=0.5).sample() == 0)
    assert(np.array_equal(
        NegativeBinomial(r=2, p=0.5).sample(3),
        np.array([7, 4, 2])
    ))

    # Test .expected_value()
    assert(NegativeBinomial(r=0, p=0).expected_value() == 0)
    assert(NegativeBinomial(r=1, p=0).expected_value() == np.inf)
    assert(NegativeBinomial(r=1, p=1).expected_value() == 0)
    assert(NegativeBinomial(r=2, p=1).expected_value() == 0)
    assert(NegativeBinomial(r=1, p=0.5).expected_value() == 1)
    assert(NegativeBinomial(r=2, p=0.5).expected_value() == 2)
    assert(NegativeBinomial(r=1, p=0.25).expected_value() == 3)
    assert(NegativeBinomial(r=2, p=0.25).expected_value() == 6)
