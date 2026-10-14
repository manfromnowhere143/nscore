"""Direction invariants and a finite exact-null control for the binary test."""

from itertools import product

import numpy as np
import pytest

from nscore.nsm import BernoulliNsmTest
from statistical_comparison_core import Decision, Hypothesis


def make_test(alternative):
    return BernoulliNsmTest(alternative=alternative, alpha=0.05, c=np.arange(2))


@pytest.mark.parametrize(
    "alternative,pairs",
    [
        (Hypothesis.P0MoreThanP1, [(1, 0), (0, 1)]),
        (Hypothesis.P0LessThanP1, [(0, 1), (1, 0)]),
    ],
)
def test_reversed_observations_preserve_valid_evidence(alternative, pairs):
    test = make_test(alternative)
    for pair in pairs:
        result = test.step(*pair)
        assert 0.0 <= test.lambda_parameter < 1.0
        assert np.isfinite(test._martingale) and test._martingale >= 0.0
        assert 0.0 <= result.info["P-Value"] <= 1.0


def test_reversing_direction_matches_swapping_streams():
    outcomes = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for sequence in product(outcomes, repeat=2):
        forward = make_test(Hypothesis.P0LessThanP1)
        reverse = make_test(Hypothesis.P0MoreThanP1)
        for a, b in sequence:
            left = forward.step(a, b)
            right = reverse.step(b, a)
            assert left.decision == right.decision
            assert forward.lambda_parameter == pytest.approx(reverse.lambda_parameter)
            assert forward._martingale == pytest.approx(reverse._martingale)
            assert left.info["P-Value"] == pytest.approx(right.info["P-Value"])


def test_reverse_exact_null_crossing_probability_through_six_pairs():
    # IID independent Bernoulli(1/2) streams: all 4**6 paths are equiprobable.
    # This guards a known finite counterexample; it is not an all-horizon proof.
    outcomes = [(0, 0), (0, 1), (1, 0), (1, 1)]
    rejected = 0
    for sequence in product(outcomes, repeat=6):
        test = make_test(Hypothesis.P0MoreThanP1)
        result = test.run_on_sequence(
            [a for a, _ in sequence], [b for _, b in sequence]
        )
        rejected += result.decision == Decision.AcceptAlternative
    assert rejected / 4**6 <= 0.05
