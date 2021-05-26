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
