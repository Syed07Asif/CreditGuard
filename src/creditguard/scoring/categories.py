"""Risk categories: bucket a `scorecard.probability_to_score` output into a
named risk band.

The bands themselves (`VERY_LOW` 750-900 / `LOW` 700-749 / `MODERATE`
650-699 / `HIGH` 550-649 / `VERY_HIGH` 300-549) are **this project's own
rules for this synthetic simulation, chosen to line up with the
recommendation policy's score thresholds** -- they are not a universal
banking-industry standard, and `config/scoring.yaml` (not this module) is
the single source of truth for where the cut points sit. See
`docs/scoring_methodology.md` for the same disclaimer in the human-facing
docs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ScoringConfigError(RuntimeError):
    """Raised when `config/scoring.yaml` describes an invalid scoring setup
    (e.g. risk bands that gap, overlap, or don't cover the full score
    range) -- a configuration bug, not a runtime scoring failure.
    """


@dataclass(frozen=True)
class RiskBand:
    """One named risk category and the inclusive score range it covers."""

    name: str
    min_score: int
    max_score: int


def load_risk_bands(config: dict[str, Any]) -> list[RiskBand]:
    """Load and validate the `risk_categories.bands` section of
    `config/scoring.yaml`, checked against `scorecard.min_score`/
    `max_score` from the same config.

    Returns the bands sorted ascending by `min_score`.

    Raises:
        ScoringConfigError: if the bands gap, overlap, or don't exactly
            cover `[scorecard.min_score, scorecard.max_score]`.
    """
    scorecard_section = config["scorecard"]
    overall_min = int(scorecard_section["min_score"])
    overall_max = int(scorecard_section["max_score"])

    bands = [
        RiskBand(
            name=entry["name"],
            min_score=int(entry["min_score"]),
            max_score=int(entry["max_score"]),
        )
        for entry in config["risk_categories"]["bands"]
    ]
    _validate_bands(bands, overall_min, overall_max)
    return sorted(bands, key=lambda band: band.min_score)


def _validate_bands(bands: list[RiskBand], overall_min: int, overall_max: int) -> None:
    if not bands:
        raise ScoringConfigError("risk_categories.bands is empty")

    for band in bands:
        if band.min_score > band.max_score:
            raise ScoringConfigError(
                f"Risk band {band.name!r} has min_score ({band.min_score}) "
                f"greater than max_score ({band.max_score})"
            )

    ordered = sorted(bands, key=lambda band: band.min_score)

    if ordered[0].min_score != overall_min:
        raise ScoringConfigError(
            f"Risk bands must start at scorecard.min_score ({overall_min}), "
            f"but the lowest band {ordered[0].name!r} starts at "
            f"{ordered[0].min_score}"
        )
    if ordered[-1].max_score != overall_max:
        raise ScoringConfigError(
            f"Risk bands must end at scorecard.max_score ({overall_max}), "
            f"but the highest band {ordered[-1].name!r} ends at "
            f"{ordered[-1].max_score}"
        )

    for previous, current in zip(ordered, ordered[1:], strict=False):
        if current.min_score == previous.max_score + 1:
            continue
        if current.min_score <= previous.max_score:
            raise ScoringConfigError(
                f"Risk bands {previous.name!r} ({previous.min_score}-"
                f"{previous.max_score}) and {current.name!r} "
                f"({current.min_score}-{current.max_score}) overlap"
            )
        raise ScoringConfigError(
            f"Gap between risk bands {previous.name!r} (ends at "
            f"{previous.max_score}) and {current.name!r} (starts at "
            f"{current.min_score}) -- scores in between are uncategorised"
        )


def categorize(score: int, bands: list[RiskBand]) -> str:
    """The name of the risk band `score` falls into.

    Raises:
        ScoringConfigError: if `score` is outside every configured band
            (should be unreachable for a `scorecard`-clipped score against
            bands validated by `load_risk_bands`, but checked explicitly
            rather than silently returning `None`).
    """
    for band in bands:
        if band.min_score <= score <= band.max_score:
            return band.name
    raise ScoringConfigError(
        f"Score {score} is not covered by any configured risk band "
        f"({[(b.name, b.min_score, b.max_score) for b in bands]})"
    )
