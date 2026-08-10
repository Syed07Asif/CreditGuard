"""Tests for creditguard.scoring.categories: risk band loading, validation
and score-to-category lookup.
"""

from __future__ import annotations

import copy

import pytest

from creditguard.scoring.categories import (
    RiskBand,
    ScoringConfigError,
    categorize,
    load_risk_bands,
)

VALID_CONFIG = {
    "scorecard": {"min_score": 300, "max_score": 900},
    "risk_categories": {
        "bands": [
            {"name": "VERY_LOW", "min_score": 750, "max_score": 900},
            {"name": "LOW", "min_score": 700, "max_score": 749},
            {"name": "MODERATE", "min_score": 650, "max_score": 699},
            {"name": "HIGH", "min_score": 550, "max_score": 649},
            {"name": "VERY_HIGH", "min_score": 300, "max_score": 549},
        ]
    },
}


def test_load_risk_bands_returns_sorted_ascending() -> None:
    bands = load_risk_bands(VALID_CONFIG)
    assert [b.name for b in bands] == [
        "VERY_HIGH",
        "HIGH",
        "MODERATE",
        "LOW",
        "VERY_LOW",
    ]


@pytest.mark.parametrize(
    "score,expected_category",
    [
        (300, "VERY_HIGH"),
        (549, "VERY_HIGH"),
        (550, "HIGH"),
        (649, "HIGH"),
        (650, "MODERATE"),
        (699, "MODERATE"),
        (700, "LOW"),
        (749, "LOW"),
        (750, "VERY_LOW"),
        (900, "VERY_LOW"),
    ],
)
def test_category_boundaries_are_exact(score: int, expected_category: str) -> None:
    bands = load_risk_bands(VALID_CONFIG)
    assert categorize(score, bands) == expected_category


def test_749_is_low_and_750_is_very_low() -> None:
    """The specific boundary pair called out in the Phase 7 acceptance criteria."""
    bands = load_risk_bands(VALID_CONFIG)
    assert categorize(749, bands) == "LOW"
    assert categorize(750, bands) == "VERY_LOW"


def test_overlapping_bands_raise_config_error() -> None:
    config = copy.deepcopy(VALID_CONFIG)
    # HIGH now overlaps MODERATE at 650-654.
    config["risk_categories"]["bands"][3]["max_score"] = 654
    with pytest.raises(ScoringConfigError, match="overlap"):
        load_risk_bands(config)


def test_gapped_bands_raise_config_error() -> None:
    config = copy.deepcopy(VALID_CONFIG)
    # Gap between VERY_HIGH (ends 549) and HIGH (now starts 555).
    config["risk_categories"]["bands"][3]["min_score"] = 555
    with pytest.raises(ScoringConfigError, match="Gap"):
        load_risk_bands(config)


def test_bands_not_starting_at_scorecard_min_raises() -> None:
    config = copy.deepcopy(VALID_CONFIG)
    config["risk_categories"]["bands"][4]["min_score"] = 310
    with pytest.raises(ScoringConfigError, match="scorecard.min_score"):
        load_risk_bands(config)


def test_bands_not_ending_at_scorecard_max_raises() -> None:
    config = copy.deepcopy(VALID_CONFIG)
    config["risk_categories"]["bands"][0]["max_score"] = 890
    with pytest.raises(ScoringConfigError, match="scorecard.max_score"):
        load_risk_bands(config)


def test_empty_bands_raises() -> None:
    config = copy.deepcopy(VALID_CONFIG)
    config["risk_categories"]["bands"] = []
    with pytest.raises(ScoringConfigError, match="empty"):
        load_risk_bands(config)


def test_inverted_band_raises() -> None:
    config = copy.deepcopy(VALID_CONFIG)
    config["risk_categories"]["bands"][0][
        "min_score"
    ] = 950  # > its own max_score (900)
    with pytest.raises(ScoringConfigError, match="greater than max_score"):
        load_risk_bands(config)


def test_categorize_raises_when_score_uncovered() -> None:
    bands = [RiskBand(name="ONLY", min_score=400, max_score=800)]
    with pytest.raises(ScoringConfigError, match="not covered"):
        categorize(300, bands)
