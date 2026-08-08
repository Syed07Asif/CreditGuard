"""Vectorized statistical sampling primitives used by the synthetic data generator.

Every function takes an explicit `numpy.random.Generator` so that callers control
reproducibility; nothing here touches global numpy random state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def truncated_normal(
    rng: np.random.Generator,
    mean: float | np.ndarray,
    std: float | np.ndarray,
    low: float | np.ndarray,
    high: float | np.ndarray,
    size: int,
) -> np.ndarray:
    """Sample a normal distribution truncated to [low, high] via rejection resampling.

    `mean`, `std`, `low` and `high` may be scalars or arrays broadcastable to `size`.
    Falls back to clipping any draws still out of range after 50 resampling rounds.
    """
    shape = (size,)
    mean_arr = np.broadcast_to(mean, shape).astype(float)
    std_arr = np.broadcast_to(std, shape).astype(float)
    low_arr = np.broadcast_to(low, shape).astype(float)
    high_arr = np.broadcast_to(high, shape).astype(float)

    values = rng.normal(mean_arr, std_arr)
    mask = (values < low_arr) | (values > high_arr)
    for _ in range(50):
        if not mask.any():
            break
        values[mask] = rng.normal(mean_arr[mask], std_arr[mask])
        mask = (values < low_arr) | (values > high_arr)
    return np.clip(values, low_arr, high_arr)


def lognormal(
    rng: np.random.Generator, mean_log: float, sigma: float, size: int
) -> np.ndarray:
    """Sample a lognormal distribution parameterised by the mean/sigma of its log."""
    return rng.lognormal(mean=mean_log, sigma=sigma, size=size)


def poisson_capped(
    rng: np.random.Generator,
    lam: float | np.ndarray,
    size: int,
    max_value: int | None = None,
) -> np.ndarray:
    """Sample a Poisson distribution, optionally capped at `max_value`."""
    values = rng.poisson(lam, size=size)
    if max_value is not None:
        values = np.clip(values, 0, max_value)
    return values


def zero_inflated_poisson(
    rng: np.random.Generator,
    zero_prob: float | np.ndarray,
    lam: float | np.ndarray,
    size: int,
) -> np.ndarray:
    """Sample a zero-inflated Poisson: structural zeros plus an ordinary Poisson."""
    zero_prob_arr = np.clip(np.broadcast_to(zero_prob, (size,)), 0.0, 1.0)
    is_structural_zero = rng.random(size) < zero_prob_arr
    poisson_draws = rng.poisson(lam, size=size)
    return np.where(is_structural_zero, 0, poisson_draws)


def sample_beta(
    rng: np.random.Generator, alpha: float, beta_param: float, size: int
) -> np.ndarray:
    """Sample from a Beta(alpha, beta) distribution."""
    return rng.beta(alpha, beta_param, size=size)


def sample_categorical(
    rng: np.random.Generator,
    categories: Sequence,
    probs: Sequence[float],
    size: int,
) -> np.ndarray:
    """Sample `size` values from `categories` per `probs` (renormalised to sum to 1)."""
    probs_arr = np.asarray(probs, dtype=float)
    probs_arr = probs_arr / probs_arr.sum()
    return rng.choice(np.asarray(categories, dtype=object), size=size, p=probs_arr)


def sample_conditional_categorical(
    rng: np.random.Generator,
    condition: np.ndarray,
    categories: Sequence,
    probs_by_condition: Mapping[object, Sequence[float]],
) -> np.ndarray:
    """Sample categories conditioned on a discrete `condition` array.

    `probs_by_condition` maps each distinct value of `condition` to a probability
    vector over `categories`.
    """
    result = np.empty(len(condition), dtype=object)
    for cond_value, probs in probs_by_condition.items():
        mask = condition == cond_value
        n = int(mask.sum())
        if n == 0:
            continue
        result[mask] = sample_categorical(rng, categories, probs, n)
    return result


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))
