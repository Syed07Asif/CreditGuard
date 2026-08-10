"""Points-to-double-odds score scaling: the standard, documented,
reproducible transform (per FR-012) from a calibrated default probability to
a 300-900 credit score, and its exact inverse.

    odds   = (1 - p_default) / p_default        # odds of NOT defaulting
    score  = OFFSET + FACTOR * ln(odds)
    FACTOR = PDO / ln(2)
    OFFSET = BASE_SCORE - FACTOR * ln(BASE_ODDS)

`BASE_SCORE`/`BASE_ODDS`/`PDO` are read from `config/scoring.yaml`'s
`scorecard` section, never hard-coded here. See
`docs/scoring_methodology.md` for the full derivation and a worked example.
Higher score = lower risk (`score` is monotonically decreasing in `p`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

# Probabilities are clipped away from the exact [0, 1] boundary before the
# odds/log transform -- p=0 or p=1 would make ln(odds) diverge to +/-inf.
# 1e-9 is tight enough to leave any realistic calibrated probability
# untouched while still handling the documented extreme-probability test
# cases (1e-9, 1 - 1e-9) without a math domain error.
PROBABILITY_EPSILON = 1e-9


@dataclass(frozen=True)
class ScorecardConfig:
    """The four scorecard tunables from `config/scoring.yaml`'s
    `scorecard` section, plus the two derived constants (`factor`/`offset`)
    every conversion uses.
    """

    base_score: float
    base_odds: float
    pdo: float
    min_score: int
    max_score: int

    @property
    def factor(self) -> float:
        """Points added for every doubling of the odds of not defaulting."""
        return self.pdo / math.log(2.0)

    @property
    def offset(self) -> float:
        """The score-scale intercept such that `probability_to_score`
        returns exactly `base_score` when the odds of not defaulting equal
        `base_odds`.
        """
        return self.base_score - self.factor * math.log(self.base_odds)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ScorecardConfig:
        """Build from the `scorecard` section of `config/scoring.yaml`."""
        section = config["scorecard"]
        return cls(
            base_score=float(section["base_score"]),
            base_odds=float(section["base_odds"]),
            pdo=float(section["pdo"]),
            min_score=int(section["min_score"]),
            max_score=int(section["max_score"]),
        )


def probability_to_score(p_default: float, config: ScorecardConfig) -> int:
    """Convert a calibrated default probability into an integer credit
    score, clipped to `[config.min_score, config.max_score]`.

    Higher score = lower risk: as `p_default` rises, the odds of not
    defaulting fall, `ln(odds)` falls, and so does the score.
    """
    p = min(max(float(p_default), PROBABILITY_EPSILON), 1.0 - PROBABILITY_EPSILON)
    odds_of_not_defaulting = (1.0 - p) / p
    raw_score = config.offset + config.factor * math.log(odds_of_not_defaulting)
    clipped = min(max(raw_score, config.min_score), config.max_score)
    return int(round(clipped))


def score_to_probability(score: float, config: ScorecardConfig) -> float:
    """The exact inverse of `probability_to_score`'s formula (before
    rounding/clipping to an integer score): recovers the default
    probability that would produce `score` on the same scale.

    Round-tripping `score_to_probability(probability_to_score(p))` is only
    exact up to the precision lost by `probability_to_score` rounding to
    the nearest integer score and clipping at the scorecard's [min, max]
    bounds -- see `tests/test_scorecard.py` for the tolerance this holds to
    in the unclipped range.
    """
    odds_of_not_defaulting = math.exp((float(score) - config.offset) / config.factor)
    p = 1.0 / (1.0 + odds_of_not_defaulting)
    return min(max(p, PROBABILITY_EPSILON), 1.0 - PROBABILITY_EPSILON)
