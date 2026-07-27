"""
Price integrity detection — rule-based, no ML.

Assesses whether the current price represents a genuine deal or a
suspicious promotion by comparing against historical statistics.

Scoring weights (per confirmed design):
  - Price deviation from historical low  : 40 %
  - Price deviation from historical avg  : 30 %
  - Marketing text risk                  : 30 %  (injected from marketing_risk.py)
"""

from __future__ import annotations

from typing import Optional

from database.queries import (
    historical_stats,
    price_deviation_from_low,
    price_deviation_from_avg,
)


def assess_price_integrity(
    history_prices: list[float],
    current_price: float,
    original_price: Optional[float] = None,
    marketing_risk_score: float = 0.0,
) -> dict:
    """Run the full price integrity assessment.

    Parameters
    ----------
    history_prices : list[float]
        Historical price data-points (oldest → newest).
    current_price : float
        The price currently shown on the product page.
    original_price : float or None
        The "original" / "list" price displayed on the page (划线价).
        Used to detect fake markdowns.
    marketing_risk_score : float
        0-100 score from :func:`analysis.marketing_risk.evaluate`.
        Injected here so the caller controls the flow.

    Returns
    -------
    dict
        {
            "risk_level": "低风险" | "中风险" | "高风险",
            "risk_score": 0-100,
            "breakdown": {...},
            "verdict": str,
        }
    """
    # ── 1. Compute historical stats ──────────────────────────────────────
    stats = historical_stats(history_prices)

    if stats["count"] < 3:
        return {
            "risk_level": "数据不足",
            "risk_score": 0,
            "breakdown": {"error": "历史数据不足（至少需要 3 条记录）"},
            "verdict": "历史数据不足，无法评估价格诚信度。请积累更多价格记录后再查询。",
        }

    # ── 2. Price-deviation scores ────────────────────────────────────────
    low_dev, low_score = price_deviation_from_low(current_price, stats)
    avg_dev, avg_score = price_deviation_from_avg(current_price, stats)

    # ── 3. Fake-original-price detection (if original_price provided) ────
    fake_original_flag = False
    fake_original_detail = ""
    if original_price and original_price > 0 and stats["highest"] > 0:
        markup_pct = round(
            (original_price - stats["highest"]) / stats["highest"] * 100, 1
        )
        if markup_pct > 20:
            fake_original_flag = True
            fake_original_detail = (
                f"标称原价 ¥{original_price:.0f} 比历史最高价 ¥{stats['highest']:.0f}"
                f" 高出 {markup_pct}%，疑似虚标原价"
            )

    # ── 4. Weighted risk score ───────────────────────────────────────────
    # low_score  * 0.4  +  avg_score * 0.3  +  marketing_risk_score * 0.3
    raw_score = (
        low_score * 0.4
        + avg_score * 0.3
        + marketing_risk_score * 0.3
    )

    # Boost if fake original price detected
    if fake_original_flag:
        raw_score = min(raw_score + 15, 100.0)

    risk_score = round(raw_score, 1)

    # ── 5. Risk level ────────────────────────────────────────────────────
    if risk_score <= 30:
        risk_level = "低风险"
    elif risk_score <= 55:
        risk_level = "中风险"
    else:
        risk_level = "高风险"

    # ── 6. Human-readable verdict ────────────────────────────────────────
    verdict = _build_verdict(
        risk_level,
        current_price,
        stats,
        low_dev,
        avg_dev,
        marketing_risk_score,
        fake_original_flag,
    )

    # ── 7. Trend direction ───────────────────────────────────────────────
    trend_pct = stats.get("trend_pct", 0.0)
    if trend_pct > 5:
        trend_label = "上涨 ↑"
    elif trend_pct < -5:
        trend_label = "下降 ↓"
    else:
        trend_label = "平稳 →"

    return {
        "risk_level": risk_level,
        "risk_score": risk_score,
        "breakdown": {
            "historical_lowest": stats["lowest"],
            "historical_average": stats["average"],
            "historical_median": stats["median"],
            "historical_highest": stats["highest"],
            "price_std": stats["std"],
            "record_count": stats["count"],
            "deviation_from_low_pct": low_dev,
            "deviation_from_low_score": low_score,
            "deviation_from_avg_z": avg_dev,
            "deviation_from_avg_score": avg_score,
            "marketing_risk_score": marketing_risk_score,
            "fake_original_price": fake_original_flag,
            "fake_original_detail": fake_original_detail,
            "trend_pct": trend_pct,
            "trend_label": trend_label,
        },
        "verdict": verdict,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _build_verdict(
    risk_level: str,
    current: float,
    stats: dict,
    low_dev: float,
    avg_dev: float,
    mkt_score: float,
    fake_original: bool,
) -> str:
    """Compose a Chinese-language verdict string."""

    parts = []

    # Price positioning
    if low_dev <= 0:
        parts.append(f"当前价格 ¥{current:.0f} 为历史新低")
    elif low_dev <= 5:
        parts.append(f"当前价格 ¥{current:.0f} 接近历史最低价 ¥{stats['lowest']:.0f}（偏离 {low_dev}%）")
    else:
        parts.append(f"当前价格 ¥{current:.0f} 比历史最低价 ¥{stats['lowest']:.0f} 高 {low_dev}%")

    # Fake original price
    if fake_original:
        parts.append("标称原价远超历史最高价，疑似虚标")

    # Marketing risk
    if mkt_score >= 60:
        parts.append("营销文案存在高风险诱导话术，请理性判断需求")
    elif mkt_score >= 30:
        parts.append("营销文案含一定诱导性词汇，注意辨别")

    # Trend
    trend = stats.get("trend_pct", 0)
    if trend > 10:
        parts.append(f"近期价格呈上涨趋势（+{trend}%），建议观望")
    elif trend < -10:
        parts.append(f"近期价格呈下降趋势（{trend}%），可继续关注")

    # Final recommendation
    if risk_level == "低风险":
        parts.append("综合评估：价格诚信，可考虑购买。")
    elif risk_level == "中风险":
        parts.append("综合评估：存在疑点，建议观望或等待更优价格。")
    else:
        parts.append("综合评估：价格异常，不建议立即购买。")

    return "。".join(parts) + "。"
