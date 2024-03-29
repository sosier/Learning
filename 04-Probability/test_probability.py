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
    NegativeBinomial, Poisson, Uniform, Normal, Exponential, Multinomial,
    Cauchy
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

    # Test .variance()
    assert(Bernoulli(p=1).variance() == 0)
    assert(Bernoulli(p=0).variance() == 0)
    assert(Bernoulli(p=0.5).variance() == 0.25)
    assert(abs(Bernoulli(p=0.8).variance() - 0.16) < 0.00000001)

    # Test .standard_deviation()
    assert(Bernoulli(p=1).standard_deviation() == 0)
    assert(Bernoulli(p=0).standard_deviation() == 0)
    assert(Bernoulli(p=0.5).standard_deviation() == np.sqrt(0.25))
    assert(abs(Bernoulli(p=0.8).standard_deviation() - np.sqrt(0.16)) < 0.00000001)

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

    # Test .variance()
    assert(Binomial(n=1, p=1).variance() == 0)
    assert(Binomial(n=1, p=0).variance() == 0)
    assert(Binomial(n=1, p=0.5).variance() == 0.25)
    assert(abs(Binomial(n=1, p=0.8).variance() - 0.16) < 0.00000001)
    assert(Binomial(n=10, p=1).variance() == 0)
    assert(Binomial(n=10, p=0).variance() == 0)
    assert(Binomial(n=10, p=0.5).variance() == 2.5)
    assert(abs(Binomial(n=10, p=0.8).variance() - 1.6) < 0.00000001)

    # Test .standard_deviation()
    assert(Binomial(n=1, p=1).standard_deviation() == 0)
    assert(Binomial(n=1, p=0).standard_deviation() == 0)
    assert(Binomial(n=1, p=0.5).standard_deviation() == np.sqrt(0.25))
    assert(abs(Binomial(n=1, p=0.8).standard_deviation() - np.sqrt(0.16)) < 0.00000001)
    assert(Binomial(n=10, p=1).standard_deviation() == 0)
    assert(Binomial(n=10, p=0).standard_deviation() == 0)
    assert(Binomial(n=10, p=0.5).standard_deviation() == np.sqrt(2.5))
    assert(abs(Binomial(n=10, p=0.8).standard_deviation() - np.sqrt(1.6)) < 0.00000001)

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

    # Test .variance()
    assert(abs(Hypergeometric(N=3, K=2, n=2).variance() - 2/9) < 0.00000001)
    assert(abs(Hypergeometric(N=3, K=2, n=1).variance() - 2/9) < 0.00000001)
    assert(Hypergeometric(N=3, K=2, n=3).variance() == 0)
    assert(Hypergeometric(N=3, K=0, n=2).variance() == 0)
    assert(Hypergeometric(N=3, K=3, n=2).variance() == 0)
    assert(Hypergeometric(N=3, K=3, n=1).variance() == 0)

    # Test .standard_deviation()
    assert(Hypergeometric(N=3, K=2, n=2).standard_deviation() == np.sqrt(2/9))
    assert(Hypergeometric(N=3, K=2, n=1).standard_deviation() == np.sqrt(2/9))
    assert(Hypergeometric(N=3, K=2, n=3).standard_deviation() == 0)
    assert(Hypergeometric(N=3, K=0, n=2).standard_deviation() == 0)
    assert(Hypergeometric(N=3, K=3, n=2).standard_deviation() == 0)
    assert(Hypergeometric(N=3, K=3, n=1).standard_deviation() == 0)

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

    # Test .variance()
    assert(Geometric(p=0).variance() == np.inf)
    assert(Geometric(p=1).variance() == 0)
    assert(Geometric(p=0.5).variance() == 2)

    # Test .standard_deviation()
    assert(Geometric(p=0).standard_deviation() == np.inf)
    assert(Geometric(p=1).standard_deviation() == 0)
    assert(Geometric(p=0.5).standard_deviation() == np.sqrt(2))

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

    # Test .variance()
    assert(NegativeBinomial(r=0, p=0).variance() == 0)
    assert(NegativeBinomial(r=1, p=0).variance() == np.inf)
    assert(NegativeBinomial(r=1, p=1).variance() == 0)
    assert(NegativeBinomial(r=2, p=1).variance() == 0)
    assert(NegativeBinomial(r=1, p=0.5).variance() == 2)
    assert(NegativeBinomial(r=2, p=0.5).variance() == 4)
    assert(NegativeBinomial(r=1, p=0.25).variance() == 12)
    assert(NegativeBinomial(r=2, p=0.25).variance() == 24)

    # Test .standard_deviation()
    assert(NegativeBinomial(r=0, p=0).standard_deviation() == 0)
    assert(NegativeBinomial(r=1, p=0).standard_deviation() == np.inf)
    assert(NegativeBinomial(r=1, p=1).standard_deviation() == 0)
    assert(NegativeBinomial(r=2, p=1).standard_deviation() == 0)
    assert(NegativeBinomial(r=1, p=0.5).standard_deviation() == np.sqrt(2))
    assert(NegativeBinomial(r=2, p=0.5).standard_deviation() == 2)
    assert(NegativeBinomial(r=1, p=0.25).standard_deviation() == np.sqrt(12))
    assert(NegativeBinomial(r=2, p=0.25).standard_deviation() == np.sqrt(24))

def test_Poisson():
    # Test .prob_of()
    assert(Poisson(lmbda=0).prob_of(0) == 1)
    assert(Poisson(lmbda=0).prob_of(1) == 0)
    assert(Poisson(lmbda=1).prob_of(0) == 1/np.exp(1))
    assert(Poisson(lmbda=1).prob_of(1) == 1/np.exp(1))
    assert(Poisson(lmbda=1).prob_of(2) == 1/(2 * np.exp(1)))
    assert(round(Poisson(lmbda=2.5).prob_of(0), 3) == 0.082)
    assert(round(Poisson(lmbda=2.5).prob_of(1), 3) == 0.205)
    assert(round(Poisson(lmbda=2.5).prob_of(2), 3) == 0.257)

    # Test .sample()
    assert(Poisson(lmbda=0).sample() == 0)
    assert(all(Poisson(lmbda=0).sample(3) == 0))

    np.random.seed(12345)
    assert(Poisson(lmbda=1).sample() == 2)
    assert(np.array_equal(
        Poisson(lmbda=1).sample(3),
        np.array([2, 0, 1])
    ))
    assert(Poisson(lmbda=5).sample() == 0)
    assert(np.array_equal(
        Poisson(lmbda=5).sample(3),
        np.array([3, 2, 4])
    ))

    # Test .expected_value()
    assert(Poisson(lmbda=0).expected_value() == 0)
    assert(Poisson(lmbda=2).expected_value() == 2)
    assert(Poisson(lmbda=3.7).expected_value() == 3.7)
    assert(Poisson(lmbda=42).expected_value() == 42)

    # Test .variance()
    assert(Poisson(lmbda=0).variance() == 0)
    assert(Poisson(lmbda=2).variance() == 2)
    assert(Poisson(lmbda=3.7).variance() == 3.7)
    assert(Poisson(lmbda=42).variance() == 42)

    # Test .standard_deviation()
    assert(Poisson(lmbda=0).standard_deviation() == 0)
    assert(Poisson(lmbda=2).standard_deviation() == np.sqrt(2))
    assert(Poisson(lmbda=3.7).standard_deviation() == np.sqrt(3.7))
    assert(Poisson(lmbda=42).standard_deviation() == np.sqrt(42))

def test_Uniform():
    # Test .prob_of()
    assert(Uniform(0, 1).prob_of() == 1)
    assert(Uniform(0, 1).prob_of(0, 1) == 1)
    assert(Uniform(-37.3, 142).prob_of() == 1)
    assert(Uniform(-37.3, 142).prob_of(-37.3, 142) == 1)
    assert(Uniform(0, 1).prob_of(0, 0.5) == 0.5)
    assert(Uniform(0, 1).prob_of(0.2, 0.5) == 0.3)
    assert(Uniform(0, 10).prob_of(0, 0.5) == 0.05)
    assert(Uniform(0, 10).prob_of(0.2, 0.5) == 0.03)

    # Test .sample()
    np.random.seed(12345)
    assert(Uniform(0, 1).sample() == 0.9296160928171479)
    assert(Uniform(2.3, 11.9).sample() == 5.337205323985145)
    assert(Uniform(-10, -1).sample() == -8.34473069490615)

    assert(np.allclose(
        Uniform(0, 1).sample(3),
        np.array([0.20456028, 0.56772503, 0.5955447])
    ))
    assert(np.allclose(
        Uniform(2.3, 11.0).sample(3),
        np.array([10.69127632, 7.98264074, 8.81548775])
    ))

    # Test .expected_value()
    assert(Uniform(0, 1).expected_value() == 0.5)
    assert(Uniform(0, 0.1).expected_value() == 0.05)
    assert(Uniform(-10, -1).expected_value() == -5.5)

    # Test .variance()
    assert(Uniform(0, 1).variance() == 1/12)
    assert(Uniform(0, 10).variance() == 100/12)
    assert(Uniform(1, 2).variance() == 1/12)
    assert(Uniform(1, 11).variance() == 100/12)

    # Test.standard_deviation()
    assert(Uniform(0, 1).standard_deviation() == np.sqrt(1/12))
    assert(Uniform(0, 10).standard_deviation() == np.sqrt(100/12))
    assert(Uniform(1, 2).standard_deviation() == np.sqrt(1/12))
    assert(Uniform(1, 11).standard_deviation() == np.sqrt(100/12))

def test_Normal():
    # Test .prob_of()
    assert(Normal(0, 1).prob_of() == 1)
    assert(Normal(0.789, 1237).prob_of() == 1)
    assert(round(Normal(0, 1).prob_of(-1, 1), 12) == 0.682689492137)
    assert(round(Normal(0, 1).prob_of(-2, 2), 12) == 0.954499736104)
    assert(round(Normal(0, 1).prob_of(-3, 3), 12) == 0.997300203937)
    assert(round(Normal(10, 1).prob_of(9, 11), 12) == 0.682689492137)
    assert(round(Normal(10, 1).prob_of(8, 12), 12) == 0.954499736104)
    assert(round(Normal(10, 1).prob_of(7, 13), 12) == 0.997300203937)
    assert(round(Normal(0, 9).prob_of(-3, 3), 12) == 0.682689492137)
    assert(round(Normal(0, 9).prob_of(-6, 6), 12) == 0.954499736104)
    assert(round(Normal(0, 9).prob_of(-9, 9), 12) == 0.997300203937)
    assert(round(Normal(10, 9).prob_of(7, 13), 12) == 0.682689492137)
    assert(round(Normal(10, 9).prob_of(4, 16), 12) == 0.954499736104)
    assert(round(Normal(10, 9).prob_of(1, 19), 12) == 0.997300203937)

    # Test .sample()
    np.random.seed(12345)
    assert(Normal(0, 1).sample() == 2.106353682693583)
    assert(Normal(0, 10).sample() == 1.9345270053213395)
    assert(Normal(10, 1).sample() == 9.26836169058641)
    assert(Normal(10, 10).sample() == 3.2945996416600973)

    assert(np.allclose(
        Normal(0, 1).sample(3),
        np.array([-0.40020881, 0.07670713, 1.02906135])
    ))
    assert(np.allclose(
        Normal(0, 10).sample(3),
        np.array([-1.91680563, -1.01832205, -5.68292916])
    ))
    assert(np.allclose(
        Normal(10, 1).sample(3),
        np.array([11.27553437, 10.58858573, 8.88631439])
    ))
    assert(np.allclose(
        Normal(10, 10).sample(3),
        np.array([18.89409371, 10.79808548, 4.68394385])
    ))

    # Test .expected_value()
    assert(Normal(0, 1).expected_value() == 0)
    assert(Normal(0, 0.1).expected_value() == 0)
    assert(Normal(-10, 11.37).expected_value() == -10)

    # Test .variance()
    assert(Normal(0, 1).variance() == 1)
    assert(Normal(0, 10).variance() == 10)
    assert(Normal(1, 2).variance() == 2)
    assert(Normal(1, 11).variance() == 11)

    # Test.standard_deviation()
    assert(Normal(0, 1).standard_deviation() == 1)
    assert(Normal(0, 100).standard_deviation() == 10)
    assert(Normal(1, 4).standard_deviation() == 2)
    assert(Normal(1, 121).standard_deviation() == 11)

def test_Exponential():
    # Test .prob_of()
    assert(Exponential(12).prob_of() == 1)
    assert(Exponential(1287456.35).prob_of() == 1)
    assert(Exponential(1).prob_of(0, 1) == (1 - np.exp(-1)))
    assert(Exponential(1).prob_of(0, 2) == (1 - np.exp(-2)))
    assert(Exponential(1).prob_of(1, 2) == (np.exp(-1) - np.exp(-2)))
    assert(Exponential(1.2).prob_of(0, 1) == (1 - np.exp(-1.2)))
    assert(Exponential(1.2).prob_of(0, 2) == (1 - np.exp(-2.4)))
    assert(
        round(Exponential(1.2).prob_of(1, 2), 10)
        == round(np.exp(-1.2) - np.exp(-2.4), 10)
    )

    # Test .sample()
    np.random.seed(12345)
    assert(Exponential(1).sample() == 2.6537906331017487)
    assert(Exponential(20.3).sample() == 0.018736284165141427)

    assert(np.allclose(
        Exponential(1).sample(3),
        np.array([0.20324143, 0.22886021, 0.83869339])
    ))
    assert(np.allclose(
        Exponential(20.3).sample(3),
        np.array([0.04459183, 0.16446461, 0.05216458])
    ))

    # Test .expected_value()
    assert(Exponential(1).expected_value() == 1)
    assert(Exponential(0.5).expected_value() == 2)
    assert(Exponential(2).expected_value() == 1/2)

    # Test .variance()
    assert(Exponential(1).variance() == 1)
    assert(Exponential(0.5).variance() == 4)
    assert(Exponential(2).variance() == 1/4)

    # Test.standard_deviation()
    assert(Exponential(1).standard_deviation() == 1)
    assert(Exponential(0.5).standard_deviation() == 2)
    assert(Exponential(2).standard_deviation() == 1/2)

def test_Multinomial():
    # Test .prob_of()
    assert Multinomial(n=1, p=np.array([1])).prob_of(np.array([1])) == 1
    assert Multinomial(n=1, p=np.array([1])).prob_of(np.array([0])) == 0
    assert (
        Multinomial(n=1, p=np.array([0.7, 0.3]))
        .prob_of(np.array([1, 0]))
        == 0.7
    )
    assert (
        Multinomial(n=1, p=np.array([0.7, 0.3]))
        .prob_of(np.array([0, 1]))
        == 0.3
    )
    assert Multinomial(n=5, p=np.array([1])).prob_of(np.array([0])) == 0
    assert Multinomial(n=5, p=np.array([1])).prob_of(np.array([4])) == 0
    assert Multinomial(n=5, p=np.array([1])).prob_of(np.array([5])) == 1
    assert Multinomial(n=5, p=np.array([1])).prob_of(np.array([6])) == 0
    assert (
        Multinomial(n=3, p=np.array([0.5, 0.5]))
        .prob_of(np.array([0, 3]))
        == 1/8
    )
    assert (
        Multinomial(n=3, p=np.array([0.5, 0.5]))
        .prob_of(np.array([1, 2]))
        == 3/8
    )
    assert (
        Multinomial(n=3, p=np.array([0.5, 0.5]))
        .prob_of(np.array([2, 1]))
        == 3/8
    )
    assert (
        Multinomial(n=3, p=np.array([0.5, 0.5]))
        .prob_of(np.array([3, 0]))
        == 1/8
    )
    assert (
        Multinomial(n=3, p=np.array([1/3, 1/3, 1/3]))
        .prob_of(np.array([0, 0, 3]))
        == Multinomial(n=3, p=np.array([1/3, 1/3, 1/3]))
        .prob_of(np.array([3, 0, 0]))
        == Multinomial(n=3, p=np.array([1/3, 1/3, 1/3]))
        .prob_of(np.array([0, 3, 0]))
        and
        abs(
            Multinomial(n=3, p=np.array([1/3, 1/3, 1/3]))
            .prob_of(np.array([0, 0, 3]))
            - 1/27
        ) <= 0.000000001
    )
    assert (
        Multinomial(n=3, p=np.array([1/3, 1/3, 1/3]))
        .prob_of(np.array([1, 1, 1]))
        == 6/27
    )
    assert (
        Multinomial(n=3, p=np.array([1/3, 1/3, 1/3]))
        .prob_of(np.array([2, 1, 0]))
        == Multinomial(n=3, p=np.array([1/3, 1/3, 1/3]))
        .prob_of(np.array([2, 0, 1]))
        == Multinomial(n=3, p=np.array([1/3, 1/3, 1/3]))
        .prob_of(np.array([0, 2, 1]))
        == Multinomial(n=3, p=np.array([1/3, 1/3, 1/3]))
        .prob_of(np.array([1, 2, 0]))
        == Multinomial(n=3, p=np.array([1/3, 1/3, 1/3]))
        .prob_of(np.array([1, 0, 2]))
        == Multinomial(n=3, p=np.array([1/3, 1/3, 1/3]))
        .prob_of(np.array([0, 1, 2]))
        == 3/27
    )
    assert (
        Multinomial(n=3, p=np.array([1/3, 1/3, 1/3]))
        .prob_of(np.array([1, 1, 2]))
        == Multinomial(n=3, p=np.array([1/3, 1/3, 1/3]))
        .prob_of(np.array([1, -1, 2]))
        == Multinomial(n=3, p=np.array([1/3, 1/3, 1/3]))
        .prob_of(np.array([0.5, 0.5, 2]))
        == 0
    )

    # Test .sample()
    assert Multinomial(n=1, p=np.array([1])).sample() == np.array([1])
    assert all(Multinomial(n=1, p=np.array([1])).sample(10) == np.array([1]))
    assert Multinomial(n=5, p=np.array([1])).sample() == np.array([5])
    assert all(Multinomial(n=5, p=np.array([1])).sample(10) == np.array([5]))

    np.random.seed(12345)  # Something up with the random seeds here...
    assert np.array_equal(
        Multinomial(n=1, p=np.array([0.7, 0.3])).sample(),
        np.array([0, 1])
    )
    assert np.array_equal(
        Multinomial(n=1, p=np.array([0.7, 0.3])).sample(3),
        np.array([
            [1, 0],
            [1, 0],
            [1, 0]
        ])
    )
    assert np.array_equal(
        Multinomial(n=3, p=np.array([0.5, 0.5])).sample(),
        np.array([0, 3])
    )
    assert np.array_equal(
        Multinomial(n=3, p=np.array([0.5, 0.5])).sample(3),
        np.array([
            [0, 3],
            [1, 2],
            [2, 1]
        ])
    )
    assert np.array_equal(
        Multinomial(n=3, p=np.array([1/4, 1/4, 1/2])).sample(),
        np.array([0, 0, 3])
    )
    assert np.array_equal(
        Multinomial(n=3, p=np.array([1/4, 1/4, 1/2])).sample(3),
        np.array([
            [0, 0, 3],
            [0, 3, 0],
            [0, 0, 3]
        ])
    )

    # Test .expected_value()
    assert Multinomial(n=1, p=np.array([1])).expected_value() == 1
    assert Multinomial(n=8, p=np.array([1])).expected_value() == 8
    assert np.array_equal(
        Multinomial(n=1, p=np.array([0.7, 0.3])).expected_value(),
        np.array([0.7, 0.3])
    )
    assert np.array_equal(
        Multinomial(n=5, p=np.array([0.7, 0.3])).expected_value(),
        np.array([3.5, 1.5])
    )
    assert np.array_equal(
        Multinomial(n=1, p=np.array([1/4, 1/4, 1/2])).expected_value(),
        np.array([1/4, 1/4, 1/2])
    )
    assert np.array_equal(
        Multinomial(n=3, p=np.array([1/4, 1/4, 1/2])).expected_value(),
        np.array([3/4, 3/4, 3/2])
    )

    # Test .variance()
    assert Multinomial(n=1, p=np.array([1])).variance() == 0
    assert Multinomial(n=8, p=np.array([1])).variance() == 0
    assert np.array_equal(
        Multinomial(n=1, p=np.array([4/10, 6/10])).variance(),
        np.array([0.24, 0.24])
    )
    assert all(
        np.abs(
            Multinomial(n=5, p=np.array([4/10, 6/10])).variance()
            - np.array([1.2, 1.2])
        ) < 0.000000000001
    )
    assert np.array_equal(
        Multinomial(n=1, p=np.array([1/4, 1/4, 1/2])).variance(),
        np.array([3/16, 3/16, 1/4])
    )
    assert np.array_equal(
        Multinomial(n=3, p=np.array([1/4, 1/4, 1/2])).variance(),
        np.array([9/16, 9/16, 3/4])
    )

    # Test .standard_deviation()
    assert Multinomial(n=1, p=np.array([1])).standard_deviation() == 0
    assert Multinomial(n=8, p=np.array([1])).standard_deviation() == 0
    assert np.array_equal(
        Multinomial(n=1, p=np.array([4/10, 6/10])).standard_deviation(),
        np.sqrt(np.array([0.24, 0.24]))
    )
    assert all(
        np.abs(
            Multinomial(n=5, p=np.array([4/10, 6/10])).standard_deviation()
            - np.sqrt(np.array([1.2, 1.2]))
        ) < 0.000000000001
    )
    assert np.array_equal(
        Multinomial(n=1, p=np.array([1/4, 1/4, 1/2])).standard_deviation(),
        np.sqrt(np.array([3/16, 3/16, 1/4]))
    )
    assert np.array_equal(
        Multinomial(n=3, p=np.array([1/4, 1/4, 1/2])).standard_deviation(),
        np.sqrt(np.array([9/16, 9/16, 3/4]))
    )

def test_Cauchy():
    # Test .prob_of()
    assert Cauchy().prob_of() == 1
    assert Cauchy(0.789, 1237).prob_of() == 1
    assert Cauchy().prob_of(-1, 1) == 0.5
    assert Cauchy(location=5).prob_of(4, 6) == 0.5
    assert Cauchy(scale=2).prob_of(-2, 2) == 0.5
    assert Cauchy(location=5, scale=2).prob_of(3, 7) == 0.5
    assert (
        round(Cauchy().prob_of(-np.sqrt(3), np.sqrt(3)), 12)
        == round(2/3, 12)
    )
    assert (
        round(
            Cauchy(location=2, scale=3).prob_of(
                3 * -np.sqrt(3) + 2,
                3 * np.sqrt(3) + 2
            ),
            12
        ) == round(2/3, 12)
    )

    # Test .sample()
    np.random.seed(12345)
    assert Cauchy().sample() == 3.443154412873596
    assert Cauchy(location=0, scale=10).sample() == 3.4504181071072697
    assert Cauchy(location=10, scale=1).sample() == 13.420477789670208
    assert Cauchy(location=10, scale=10).sample() == 8.851760549075378

    assert np.allclose(
        Cauchy().sample(3),
        np.array([1.61365108, 47.42837438, 2.62438296])
    )
    assert np.allclose(
        Cauchy(location=0, scale=10).sample(3),
        np.array([-45.05730325, 4.65096137, -0.2699795])
    )
    assert np.allclose(
        Cauchy(location=10, scale=1).sample(3),
        np.array([11.03625118, 4.69336972, 11.33160151])
    )
    assert np.allclose(
        Cauchy(location=10, scale=10).sample(3),
        np.array([17.78125186, 8.3692884, 20.49814144])
    )

    # Test .expected_value()
    assert np.isnan(Cauchy().expected_value())
    assert np.isnan(Cauchy(-10, 11.37).expected_value())

    # Test .variance()
    assert np.isnan(Cauchy().variance())
    assert np.isnan(Cauchy(-10, 11.37).variance())

    # Test.standard_deviation()
    assert np.isnan(Cauchy().standard_deviation())
    assert np.isnan(Cauchy(-10, 11.37).standard_deviation())
