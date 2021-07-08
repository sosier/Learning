import numpy as np

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

def simulate_birthday_problem(num_people, num_simulations=1000):
    results = np.array([])
    for _ in range(num_simulations):
        birthdays = np.random.choice(365, size=num_people)
        birthday_counts = np.unique(birthdays, return_counts=True)[-1]
        results = np.append(results, any(birthday_counts > 1))

    return np.mean(results)

def probability_two_aces_info_comparison():
    # 1. Let the deck be an array length 52
    # 2. Let the first 4 cards be the Aces (i = 0 to 3), with the very first
    #    card (i = 0) being the Ace of Spades
    # 3. Let all possible two card hands be defined by the 52 x 52 matrix
    #    where the first card dealt is the rows and the second the columns
    all_hands = np.ones((52, 52))  # Where 1 = Valid Hand
    for i in range(52):
        # Diagonal (same card twice) is impossible and so must be zeroed out:
        all_hands[i][i] = 0

    # First, what's the probability of a hand with 2 Aces
    # GIVEN the hand has at least one Ace
    hands_with_one_ace = all_hands.copy()
    hands_with_one_ace[4:, 4:] = 0  # These hands have NO Aces
    num_hands_with_one_ace = hands_with_one_ace.sum()

    hands_with_two_aces = hands_with_one_ace.copy()
    hands_with_two_aces[4:] = 0  # Zero if first card NOT Ace
    hands_with_two_aces[:, 4:] = 0  # Zero if second card NOT Ace
    num_hands_with_two_aces = hands_with_two_aces.sum()

    prob_two_aces_given_one_ace = (
        num_hands_with_two_aces / num_hands_with_one_ace  # Will be 1/33
    )

    # Second compare to the probability of a hand with 2 Aces
    # GIVEN one of the cards is the Ace of Spades (card i = 0)
    hands_with_ace_of_spades = all_hands.copy()
    hands_with_ace_of_spades[1:, 1:] = 0  # These hands have NO Ace of Spades
    num_hands_with_ace_of_spades = hands_with_ace_of_spades.sum()

    hands_with_two_aces = hands_with_ace_of_spades.copy()
    hands_with_two_aces[4:] = 0  # Zero if first card NOT Ace
    hands_with_two_aces[:, 4:] = 0  # Zero if second card NOT Ace
    num_hands_with_two_aces = hands_with_two_aces.sum()

    prob_two_aces_given_ace_of_spades = (
        num_hands_with_two_aces / num_hands_with_ace_of_spades  # Will be 1/17
    )

    return {
        "prob_two_aces_given_one_ace": prob_two_aces_given_one_ace,
        "prob_two_aces_given_ace_of_spades": prob_two_aces_given_ace_of_spades
    }

def simulate_probability_has_disease_given_positive_medical_test(
        disease_prevalence=0.01, test_accuracy=0.95, num_simulations=1000
    ):
    has_disease = (
        np.random.rand(num_simulations) <= disease_prevalence
    ).astype(int)
    test_accurate = (
        np.random.rand(num_simulations) <= test_accuracy
    ).astype(int)

    # Pull out positive tests:
    num_has_disease_and_test_accurate = (
        has_disease & test_accurate
    ).astype(int).sum()
    num_no_disease_and_test_inaccurate = (
        (has_disease == 0) & (test_accurate == 0)
    ).astype(int).sum()

    return (
        num_has_disease_and_test_accurate
        /
        (num_has_disease_and_test_accurate + num_no_disease_and_test_inaccurate)
    )

def simulate_monty_hall_problem(num_simulations=1000):
    """
    Returns: Probability of winning if you switch
    """
    results = []
    for _ in range(num_simulations):
        doors = np.arange(3)
        door_with_car = np.random.choice(doors)
        door_you_picked = np.random.choice(doors)

        doors_you_didnt_pick = np.delete(doors, door_you_picked)
        doors_monty_can_open = np.delete(doors, [door_you_picked, door_with_car])
        door_monty_opens = np.random.choice(doors_monty_can_open)

        door_you_can_switch_to = np.delete(doors, [door_you_picked, door_monty_opens])[0]
        result_of_switching = int(door_you_can_switch_to == door_with_car)
        results.append(result_of_switching)

    return np.mean(results)

class Bernoulli():
    def __init__(self, p):
        """
        p = Probability of 1
        1 - p = q = Probability of 0
        """
        assert(0 <= p <= 1)
        self.p = p
        self.q = 1 - self.p

    def sample(self, n=1):
        assert(n >= 1)
        result = (np.random.rand(n) <= self.p).astype(int)
        return result if n > 1 else result[0]

    def prob_of(self, k):
        if k not in [0, 1]:
            return 0
        elif k == 1:
            return self.p
        else:
            return self.q

    def expected_value(self):
        return self.p

    def variance(self):
        """
        Derived by E( (X - E(X))**2 ) = E(X**2) - E(X)**2, then calculating and
        filling in E(X**2) and E(X)**2 and simplifying:

          E(X**2) = sum([ x**2 * P(X = x) for all x])       # By LOTUS
            x can only be 0 or 1 so:
                  = 0**2 * P(X = 0) + 1**2 * P(X = 1)
                  = P(X = 1)
                  = p

          E(X) = p
          E(X)**2 = p**2

          E(X**2) - E(X)**2 = p - p**2
                            = p * (1 - p)
                            = p * q

        """
        return self.p * self.q

    def standard_deviation(self):
        return np.sqrt(self.variance())


class Binomial():
    def __init__(self, n, p):
        """
        n = # of trials

        For any given trial:
            p = Probability of 1
            1 - p = Probability of 0
        """
        assert(n >= 0 and type(n) == int)
        self.n = n

        assert(0 <= p <= 1)
        self.p = p
        self.q = 1 - self.p

    def sample(self, num_samples=1):
        assert(num_samples >= 1)
        result = (np.random.rand(num_samples, self.n) <= self.p).astype(int)
        # Count the # of success for of the samples
        result = np.sum(result, axis=1)  # Sum the column value for each row
        return result if num_samples > 1 else result[0]

    def prob_of(self, k):
        if k < 0 or k > self.n or type(k) != int:
            return 0
        else:
            return n_choose_k(self.n, k) * self.p**k * self.q**(self.n - k)

    def expected_value(self):
        return self.n * self.p

    def variance(self):
        """
        Derivation:
         1. Binomial = sum of n i.i.d. independent identically distributed
            Bernoulli(p)
         2. Var(X + Y) = Var(X) + Var(Y) if X and Y and independent
         3. From 1 & 2:
            Var(Binomial(n, p)) = n * Var(Bernoulli(p))
                                = n * p * q
        """
        return self.n * self.p * self.q

    def standard_deviation(self):
        return np.sqrt(self.variance())

class Hypergeometric():
    def __init__(self, N, K, n):
        """
        N = Size of overall population (int >= 0)
        K = Total size of "success" draws (int >=0 and <= N)
        n = # of samples without replacement (int >=0 and <= N)
        """
        assert(N >= 0 and type(n) == int)
        self.N = N

        assert(0 <= K <= self.N and type(K) == int)
        self.K = K

        assert(0 <= n <= self.N and type(n) == int)
        self.n = n

        self.data_set_for_sampling = np.append(
            np.ones(self.K),
            np.zeros(self.N - self.K)
        )

    def sample(self, num_samples=1):
        assert(num_samples >= 1)
        result = np.array([
            # Sum counts # of success a.k.a. 1's
            np.sum(np.random.choice(
                self.data_set_for_sampling,
                size=self.n,
                replace=False
            ))
            for _ in range(num_samples)
        ])

        return result if num_samples > 1 else result[0]

    def prob_of(self, k):
        if k < 0 or k > self.K or k > self.n or type(k) != int:
            return 0
        else:
            return (
                n_choose_k(self.K, k) * n_choose_k(self.N - self.K, self.n - k)
                / n_choose_k(self.N, self.n)
            )

    def expected_value(self):
        return self.n * (self.K/ self.N)

    def variance(self):
        pass

    def standard_deviation(self):
        return np.sqrt(self.variance())

class Geometric():
    def __init__(self, p):
        """
        Geometric = # of "failures" before a "success" (e.g. for coin flipping,
            the number of tails before getting a heads)

        p = Probability of "success" on any given i.i.d. (independent
            identically distributed) trial
        """
        assert(0 <= p <= 1)
        self.p = p
        self.q = 1 - p

    def _sample_one(self):
        if self.p == 0:
            return np.inf
        elif self.p == 1:
            return 0
        else:
            count = 0
            random_draw = np.random.rand()
            while random_draw > self.p:
                count += 1
                random_draw = np.random.rand()

            return count

    def sample(self, num_samples=1):
        assert(num_samples >= 1)
        result = np.array([
            self._sample_one()
            for _ in range(num_samples)
        ])

        return result if num_samples > 1 else result[0]

    def prob_of(self, k):
        if k < 0 or type(k) != int:
            return 0
        elif k == 0 and self.p == 1:
            return 1
        else:
            return self.q**k * self.p

    def expected_value(self):
        if self.p == 0:
            return np.inf
        else:
            return self.q / self.p

    def variance(self):
        """
        Derived by E( (X - E(X))**2 ) = E(X**2) - E(X)**2, then calculating and
        filling in E(X**2) and E(X)**2 and simplifying:

          E(X**2) = sum([ k**2 * P(X = k) for all k])       # By LOTUS
            k can be 0 to infinity:
                  = sum_to_infinity([k**2 * q**k * p])
                  = p * sum_to_infinity([k**2 * q**k])

            sum_to_infinity([k**2 * q**k]) = q * (1 + q) / (1 - q)**3
            (from https://en.wikipedia.org/wiki/List_of_mathematical_series#Low-order_polylogarithms
            --> see Li_-2(z))

                  = p * q * (1 + q) / (1 - q)**3

            Because q = 1 - p, 1 - q = p:
                  = p * q * (1 + q) / p**3
                  = q * (1 + q) / p**2

          E(X) = q / p
          E(X)**2 = q**2 / p**2

          E(X**2) - E(X)**2 = q * (1 + q) / p**2 - q**2 / p**2
                            = (q + q**2 - q**2) / p**2
                            = q / p**2
        """
        if self.p == 0:
            return np.inf
        else:
            return self.q / self.p**2

    def standard_deviation(self):
        return np.sqrt(self.variance())


class NegativeBinomial():
    def __init__(self, r, p):
        """
        Given i.i.d. Bernoulli(p) trials, # of failures before the r'th success

        r = # of successes (int >= 0)
        p = Probability of success for any given trial (float in range [0, 1])
        """
        assert(r >= 0 and type(r) == int)
        self.r = r

        assert(0 <= p <= 1)
        self.p = p
        self.q = 1 - self.p

    def _sample_one(self):
        if self.r == 0:
            return 0
        elif self.p == 0:
            return np.inf
        elif self.p == 1:
            return 0
        else:
            successes = 0
            failures = 0

            while successes < self.r:
                random_draw = np.random.rand()

                if random_draw > self.p:
                    failures += 1
                else:
                    successes += 1

            return failures

    def sample(self, num_samples=1):
        """
        num_samples = # of samples to perform (int >= 1)
        """
        assert(num_samples >= 1 and type(num_samples) == int)
        result = np.array([
            self._sample_one()
            for _ in range(num_samples)
        ])

        return result if num_samples > 1 else result[0]

    def prob_of(self, n):
        if self.p == 0 and n == np.inf:
            return 1
        elif n < 0 or type(n) != int:
            return 0
        elif self.r == 0:
            if n == 0:
                return 1
            else:
                return 0
        else:
            return (
                # -1 since last trial is always success, other trials can be in
                # any order:
                n_choose_k(n + self.r - 1, self.r - 1)
                * self.p**self.r
                * self.q**n
            )

    def expected_value(self):
        if self.r == 0 or self.p == 1:
            return 0
        elif self.p == 0:
            return np.inf
        else:
            return self.r * self.q / self.p

    def variance(self):
        pass

    def standard_deviation(self):
        return np.sqrt(self.variance())

class Poisson():
    def __init__(self, lmbda):
        """
        Count of "successes" in a fixed time period (e.g. day, month, year,
        etc.) given some average rate of occurance (lambda), e.g. 3 per day on
        average

        Also equal to the limit of the Binomial distribution as n apporaches
        infinity and p for any given n approaches 0

        lmbda = lambda shortened b/c lambda is a reserved word in Python; as
            above the average # of successes in the time period ("rate")
            (float >= 0)
        """
        assert(lmbda >= 0)
        self.lmbda = lmbda

    def sample(self, num_samples=1, accuracy=1000):
        """
        Returns an approximate Poisson sample using the Binomial distribution
        with a large n and small p

        num_samples = # of samples to perform (int >= 1)
        accuracy = parameter controlling how precise the approximation should be
            (higher = more accurate; defaults 1000 or n = round(1000 x lambda)
            and p = 1/1000)
        """
        if self.lmbda == 0:
            return Binomial(0, 0).sample(num_samples)

        n = round(self.lmbda * accuracy)
        p = self.lmbda / n
        return Binomial(n, p).sample(num_samples)

    def prob_of(self, k):
        """
        Derived by the limit of the PMF (`prob_of()`) of `Binomial(n, p)` as n
        apporaches infinity and p apporaches 0
        """
        assert(k >= 0 and type(k) == int)
        return (
            (self.lmbda**k / factorial(k))
            * np.exp(-self.lmbda)
        )

    def expected_value(self):
        return self.lmbda  # by definition

    def variance(self):
        """
        Derived by E( (X - E(X))**2 ) = E(X**2) - E(X)**2, then calculating and
        filling in E(X**2) and E(X)**2 and simplifying:

          E(X**2) = sum([ k**2 * P(X = k) for all k])       # By LOTUS
            k can be 0 to infinity:
                  = sum_to_infinity([k**2 * e**-lambda * lambda**k / k!])
                  = e**-lambda * sum_to_infinity([k**2 * lambda**k / k!])

            Starting with Taylor series for e**x:
              sum_to_infinity([lambda**k / k!]) = e**lambda
            1. Take derivative of each side with respect to lambda:
              sum_to_infinity([k * lambda**(k-1) / k!]) = e**lambda
            2. Multipy each side by lambda ("replinish the lambda"):
              sum_to_infinity([k * lambda**k / k!]) = lambda * e**lambda
            3. Take derivative of each side with respect to lambda:
              sum_to_infinity([k**2 * lambda**(k-1) / k!]) = lambda * e**lambda + e**lambda
                                                           = e**lambda * (lambda + 1)
            4. Multipy each side by lambda ("replinish the lambda"):
              sum_to_infinity([k**2 * lambda**k / k!]) = e**lambda * (lambda + 1) * lambda

          (continuing E(X**2)...)
                  = e**-lambda * sum_to_infinity([k**2 * lambda**k / k!])
            From #4 above:
                  = e**-lambda * e**lambda * (lambda + 1) * lambda
                  = (lambda + 1) * lambda
                  = lambda**2 + lambda

          E(X) = lambda
          E(X)**2 = lambda**2

          E(X**2) - E(X)**2 = lambda**2 + lambda - lambda**2
                            = lambda
        """
        return self.lmbda

    def standard_deviation(self):
        return np.sqrt(self.variance())

class Uniform():
    def __init__(self, start, end):
        """
        Simply a continuous distribution where all values in some range are
        equally likely (or more precisely all equally sized ranges of values in
        some larger range are equally likely)

        start = Start of the range (numeric)
        end = End of the range (numeric)
        """
        assert(-np.inf < float(start) < float(end) < np.inf)
        self.start = start
        self.end = end
        self.range = self.end - self.start

    def sample(self, num_samples=1):
        """
        num_samples = # of samples to perform (int >= 1)
        """
        assert(num_samples >= 1 and type(num_samples) == int)
        result = np.random.rand(num_samples)
        result = (result * self.range) + self.start

        return result if num_samples > 1 else result[0]

    def prob_of(self, start=-np.inf, end=np.inf):
        assert(start <= end)

        if start == -np.inf and end == np.inf:
            return 1
        elif start <= self.start and end >= self.end:
            return 1
        elif start > self.end or end < self.start:
            return 0
        else:
            if start <= self.start:
                start = self.start

            if end >= self.end:
                end = self.end

            return (end - start) / self.range

    def expected_value(self):
        return self.start + self.range / 2

    def variance(self):
        """
        Derived by E( (X - E(X))**2 ) = E(X**2) - E(X)**2, then calculating and
        filling in E(X**2) and E(X)**2 and simplifying (full calcuations too
        long to include here)
        """
        return self.range**2 / 12

    def standard_deviation(self):
        return np.sqrt(self.variance())

class Normal():
    def __init__(self, mu=0, sigma_squared=1):
        """
        mu = mean of the distribution (numeric)
        sigma_squared = variance (numeric >= 0)
        """
        assert(sigma_squared >= 0)
        self.mu = mu
        self.sigma_squared = sigma_squared

    def sample(self, num_samples=1, precision=1000):
        """
        num_samples = # of samples to perform (int >= 1)
        precision = # of samples to use to construct the generating
            approximately normal distribution (higher = more accurate)
        """
        assert(num_samples >= 1 and type(num_samples) == int)

        # Generate an appoximately normal distribution using the sum of 30
        # repeated random Uniform(0, 1):
        raw_data = np.random.rand(precision, 30)
        raw_data = np.sum(raw_data, axis=1)
        appox_normal_mean = np.mean(raw_data)
        appox_normal_standard_deviation = np.std(raw_data)

        # Repeat the same process to draw our "result" sample from the
        # approximately normal distribution:
        result = np.random.rand(num_samples, 30)
        result = np.sum(result, axis=1)

        # Calculate appoximate normal result to standard normal result:
        result = (result - appox_normal_mean) / appox_normal_standard_deviation

        # Convert to result for this particular Normal:
        result = result * np.sqrt(self.sigma_squared) + self.mu

        return result if num_samples > 1 else result[0]

    def __PMF(self, x):
        return np.exp(-x**2/2) / np.sqrt(2 * np.pi)

    def __sub_F_of_PMF(self, z, n):
        """
        Antiderivative of (-1/2)**n * z**2n / n! with respect to z
        """
        return (-1/2)**n * z**(2*n + 1)/(2*n + 1) / factorial(n) # + c

    def prob_of(self, start=-np.inf, end=np.inf, precision=30):
        """
        No closed form "indefinite" integral of this can be calculated so we
        have to approximate it using the Taylor series for e**x:
            e**x = sum_to_infinity(x**n / n!)

        N(z) = 1 / sqrt(2 * pi) * e**(-z**2/2)
        N(z) = 1 / sqrt(2 * pi) * sum_to_infinity(x**n / n!)
            ...where x = -z**2/2
        N(z) = 1 / sqrt(2 * pi) * sum_to_infinity((-z**2/2)**n / n!)

        integral_a_to_b(N(z))
            = 1 / sqrt(2 * pi) * sum_to_infinity(integral_a_to_b((-z**2/2)**n / n!))
            = 1 / sqrt(2 * pi) * sum_to_infinity(integral_a_to_b((-1/2)**n * z**2n / n!))
            = 1 / sqrt(2 * pi) * sum_to_infinity(F(b, n) - F(a, n))
                ...where antiderivative F(x, n) = (-1/2)**n * x**(2n + 1)/(2n + 1) / n! + c
        """
        assert(start <= end)

        if start == -np.inf and end == np.inf:
            return 1
        else:
            # Standardize start and stop:
            sigma = np.sqrt(self.sigma_squared)
            start = (start - self.mu) / sigma
            end = (end - self.mu) / sigma

            return 1/np.sqrt(2 * np.pi) * np.sum([
                self.__sub_F_of_PMF(end, n) - self.__sub_F_of_PMF(start, n)
                for n in range(0, precision)
            ])

    def expected_value(self):
        return self.mu  # By definition

    def variance(self):
        return self.sigma_squared  # By definition

    def standard_deviation(self):
        return np.sqrt(self.variance())

print(sorted(Normal(0, 1).sample(100)))
