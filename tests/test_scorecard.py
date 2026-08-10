"""Tests for creditguard.scoring.scorecard: the points-to-double-odds
credit score transform and its inverse.
"""

from __future__ import annotations

import math

import pytest

from creditguard.scoring.scorecard import (
    PROBABILITY_EPSILON,
    ScorecardConfig,
    probability_to_score,
    score_to_probability,
)


@pytest.fixture
def config() -> ScorecardConfig:
    """The project's documented defaults: base_score=600, base_odds=20:1, PDO=40."""
    return ScorecardConfig(
        base_score=600, base_odds=20.0, pdo=40.0, min_score=300, max_score=900
    )


def test_from_config_reads_scorecard_section() -> None:
    cfg = ScorecardConfig.from_config(
        {
            "scorecard": {
                "base_score": 600,
                "base_odds": 20.0,
                "pdo": 40.0,
                "min_score": 300,
                "max_score": 900,
            }
        }
    )
    assert cfg.base_score == 600
    assert cfg.base_odds == 20.0
    assert cfg.pdo == 40.0


def test_factor_and_offset_derived_correctly(config: ScorecardConfig) -> None:
    expected_factor = 40.0 / math.log(2.0)
    assert config.factor == pytest.approx(expected_factor)
    expected_offset = 600.0 - expected_factor * math.log(20.0)
    assert config.offset == pytest.approx(expected_offset)


def test_base_odds_produces_base_score_exactly(config: ScorecardConfig) -> None:
    """At p such that odds-of-not-defaulting == base_odds (20:1, i.e.
    p = 1/21), the score should land exactly on base_score.
    """
    p_at_base_odds = 1.0 / (1.0 + config.base_odds)
    assert probability_to_score(p_at_base_odds, config) == round(config.base_score)


def test_hand_computed_worked_example(config: ScorecardConfig) -> None:
    """p = 0.087, matching docs/scoring_methodology.md's worked example."""
    p = 0.087
    odds = (1 - p) / p
    raw_score = config.offset + config.factor * math.log(odds)
    assert probability_to_score(p, config) == round(raw_score)
    assert probability_to_score(p, config) == 563


@pytest.mark.parametrize(
    "p_default,expected_score",
    [
        (0.01, 692),
        (0.02, 652),
        (0.05, 597),
        (0.10, 554),
        (0.20, 507),
        (0.30, 476),
        (0.50, 427),
    ],
)
def test_hand_computed_lookup_table(
    config: ScorecardConfig, p_default: float, expected_score: int
) -> None:
    """The lookup table published in docs/scoring_methodology.md."""
    assert probability_to_score(p_default, config) == expected_score


def test_doubling_odds_is_worth_exactly_pdo_points(config: ScorecardConfig) -> None:
    """A doubling of the odds of not defaulting should move the raw
    (unclipped, unrounded) score by exactly PDO points -- the defining
    property of the transform.
    """
    p_low = 0.10
    odds_low = (1 - p_low) / p_low
    odds_high = odds_low * 2.0
    p_high = 1.0 / (1.0 + odds_high)

    raw_low = config.offset + config.factor * math.log(odds_low)
    raw_high = config.offset + config.factor * math.log(odds_high)
    assert raw_high - raw_low == pytest.approx(config.pdo)
    # p_high < p_low (odds of not defaulting doubled means default became
    # less likely), so this should also hold via the public function.
    assert probability_to_score(p_high, config) > probability_to_score(p_low, config)


def test_monotonicity_higher_probability_never_yields_higher_score(
    config: ScorecardConfig,
) -> None:
    probabilities = sorted(
        {
            0.0001,
            0.001,
            0.005,
            0.01,
            0.02,
            0.05,
            0.1,
            0.15,
            0.2,
            0.3,
            0.4,
            0.5,
            0.7,
            0.9,
            0.99,
        }
    )
    scores = [probability_to_score(p, config) for p in probabilities]
    for lower_score, higher_score in zip(scores, scores[1:], strict=False):
        assert higher_score <= lower_score


@pytest.mark.parametrize("p_extreme", [1e-9, 1.0 - 1e-9])
def test_clipping_at_extreme_probabilities(
    config: ScorecardConfig, p_extreme: float
) -> None:
    score = probability_to_score(p_extreme, config)
    assert config.min_score <= score <= config.max_score
    if p_extreme < 0.5:
        assert score == config.max_score
    else:
        assert score == config.min_score


def test_clipping_beyond_the_epsilon_boundary_does_not_raise(
    config: ScorecardConfig,
) -> None:
    """Exactly 0.0 or 1.0 would otherwise hit a math-domain error in ln();
    PROBABILITY_EPSILON guards both ends.
    """
    assert probability_to_score(0.0, config) == config.max_score
    assert probability_to_score(1.0, config) == config.min_score


@pytest.mark.parametrize(
    "p_default", [0.001, 0.01, 0.03, 0.05, 0.087, 0.1, 0.2, 0.3, 0.5]
)
def test_round_trip_within_tolerance(config: ScorecardConfig, p_default: float) -> None:
    """score_to_probability(probability_to_score(p)) should recover p
    within a small relative tolerance -- the only error source is
    probability_to_score's rounding to the nearest integer score (clipping
    doesn't kick in for any of these moderate probabilities).
    """
    score = probability_to_score(p_default, config)
    assert config.min_score < score < config.max_score  # confirms not clipped
    recovered = score_to_probability(score, config)
    assert recovered == pytest.approx(p_default, rel=0.01)


def test_score_to_probability_is_the_algebraic_inverse_pre_rounding(
    config: ScorecardConfig,
) -> None:
    """Without the intermediate round-to-int step, the transform and its
    inverse should compose to the identity to floating-point precision.
    """
    p = 0.123456
    odds = (1 - p) / p
    raw_score = config.offset + config.factor * math.log(odds)
    recovered = score_to_probability(raw_score, config)
    assert recovered == pytest.approx(p, rel=1e-9)


def test_score_to_probability_stays_in_open_unit_interval(
    config: ScorecardConfig,
) -> None:
    for score in (config.min_score, config.base_score, config.max_score):
        p = score_to_probability(score, config)
        assert PROBABILITY_EPSILON <= p <= 1.0 - PROBABILITY_EPSILON
