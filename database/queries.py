"""
Statistical queries over price history.

Provides the numbers that feed into the price-integrity and marketing-risk
analysers.  Works on any list of price floats — storage-format agnostic.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple


def historical_stats(prices: List[float]) -> dict:
    """Compute core statistics for a price history.

    Parameters
    ----------
    prices : list of float
        Historical prices, sorted oldest → newest.

    Returns
    -------
    dict with keys:
        count, lowest, highest, average, median, std,
        recent_avg (last 7), trend_pct
    """
    if not prices:
        return _empty_stats()

    n = len(prices)
    avg = sum(prices) / n

    # Standard deviation (population formula is fine for our use-case)
    variance = sum((p - avg) ** 2 for p in prices) / n
    std = math.sqrt(variance)

    # Median
    sorted_prices = sorted(prices)
    mid = n // 2
    if n % 2 == 0:
        median = (sorted_prices[mid - 1] + sorted_prices[mid]) / 2
    else:
        median = sorted_prices[mid]

    # Recent 7-day average (or all if less than 7)
    recent = prices[-min(7, n) :]
    recent_avg = sum(recent) / len(recent)

    # Trend: compare recent 7-day avg vs all-time avg
    if avg > 0 and len(recent) >= 3:
        trend_pct = round((recent_avg - avg) / avg * 100, 1)
    else:
        trend_pct = 0.0

    return {
        "count": n,
        "lowest": round(min(prices), 2),
        "highest": round(max(prices), 2),
        "average": round(avg, 2),
        "median": round(median, 2),
        "std": round(std, 2),
        "recent_avg": round(recent_avg, 2),
        "trend_pct": trend_pct,
    }


def price_deviation_from_low(current_price: float, stats: dict) -> Tuple[float, float]:
    """How far is *current_price* above the historical lowest?

    Returns (deviation_pct, score_0_100).

    deviation_pct = (current - lowest) / lowest * 100
    score = clamped mapping of deviation_pct → 0..100
    """
    lowest = stats.get("lowest", 0)
    if lowest <= 0:
        return (0.0, 0.0)

    deviation = round((current_price - lowest) / lowest * 100, 1)

    # Score mapping: 0% deviation → 0 score, 30%+ → 100 score
    if deviation <= 0:
        score = 0.0
    elif deviation >= 30:
        score = 100.0
    else:
        score = round(deviation / 30 * 100, 1)

    return (deviation, score)


def price_deviation_from_avg(current_price: float, stats: dict) -> Tuple[float, float]:
    """How far is *current_price* above the historical average, in std units?

    Returns (z_score, score_0_100).
    """
    avg = stats.get("average", 0)
    std = stats.get("std", 0)

    if avg <= 0:
        return (0.0, 0.0)

    # Use std if meaningful; otherwise fall back to pct deviation
    if std > 0 and std < avg * 0.5:  # std is reasonable (not noise, not extreme)
        z = round((current_price - avg) / std, 2)
    else:
        # Fallback: percentage deviation from average
        z = round((current_price - avg) / avg * 100, 1)

    # Score mapping based on z-score
    #   z ≤ -1   → 0   (genuinely below average)
    #   z = 0    → 30  (at average)
    #   z ≥ 2    → 100 (significantly above average)
    if z <= -1:
        score = 0.0
    elif z <= 0:
        score = round(30 * (z + 1) / 1, 1)  # linear from 0→30
    elif z <= 2:
        score = round(30 + 70 * z / 2, 1)    # linear from 30→100
    else:
        score = 100.0

    return (z, score)


def _empty_stats() -> dict:
    return {
        "count": 0,
        "lowest": 0,
        "highest": 0,
        "average": 0,
        "median": 0,
        "std": 0,
        "recent_avg": 0,
        "trend_pct": 0.0,
    }
