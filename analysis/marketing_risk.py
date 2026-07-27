"""
Marketing rhetoric risk analyser — keyword-based, no ML.

Categorises promotional language into three risk dimensions:
  1. Time pressure (时间诱导)   — "最后一天", "限时", "倒计时" ...
  2. Scarcity induction (稀缺诱导) — "仅剩", "已抢", "库存告急" ...
  3. Price anchoring (价格诱导)   — "原价", "打骨折", "血亏" ...

Returns a 0-100 risk score and a structured breakdown.
"""

from __future__ import annotations

from typing import List, Dict

# ──────────────────────────────────────────────────────────────────────────────
# Keyword library (~50 entries across 3 categories)
# ──────────────────────────────────────────────────────────────────────────────

# Each entry: (keyword, weight) — weight 1-3, higher = stronger manipulation
# Weights: 1 = mild hint, 2 = moderate pressure, 3 = aggressive manipulation

TIME_PRESSURE_KEYWORDS: List[tuple] = [
    # Aggressive (weight 3)
    ("最后一天", 3),
    ("最后几小时", 3),
    ("即将结束", 3),
    ("最后机会", 3),
    ("错过今天再等一年", 3),
    ("仅此一天", 3),
    ("最后疯抢", 3),
    ("倒计时", 3),
    ("即将涨价", 3),
    # Moderate (weight 2)
    ("限时优惠", 2),
    ("限时特惠", 2),
    ("限时抢购", 2),
    ("限时折扣", 2),
    ("今日特价", 2),
    ("活动倒计时", 2),
    ("手慢无", 2),
    ("马上抢", 2),
    # Mild (weight 1)
    ("今日特惠", 1),
    ("限时活动", 1),
    ("抓紧时间", 1),
    ("促销倒计时", 1),
]

SCARCITY_KEYWORDS: List[tuple] = [
    # Aggressive (weight 3)
    ("仅剩", 3),
    ("最后一件", 3),
    ("已抢光", 3),
    ("库存告急", 3),
    ("限量发售", 3),
    ("每人限购", 3),
    ("即将售罄", 3),
    ("先到先得", 3),
    # Moderate (weight 2)
    ("已抢", 2),
    ("热销", 2),
    ("爆款", 2),
    ("抢购中", 2),
    ("余量不多", 2),
    ("数量有限", 2),
    ("售完即止", 2),
    # Mild (weight 1)
    ("热卖", 1),
    ("畅销", 1),
    ("人气爆款", 1),
    ("限量", 1),
    ("手慢无", 1),
]

PRICE_ANCHORING_KEYWORDS: List[tuple] = [
    # Aggressive (weight 3)
    ("打骨折", 3),
    ("血亏", 3),
    ("亏本清仓", 3),
    ("跳楼价", 3),
    ("白菜价", 3),
    ("不要钱", 3),
    ("白送", 3),
    # Moderate (weight 2)
    ("原价", 2),
    ("吊牌价", 2),
    ("跌破底价", 2),
    ("历史最低", 2),
    ("全网最低", 2),
    ("史低价", 2),
    ("抄底价", 2),
    ("跌破", 2),
    # Mild (weight 1)
    ("折扣", 1),
    ("降价", 1),
    ("优惠", 1),
    ("特价", 1),
    ("划算", 1),
    ("性价比", 1),
]


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def evaluate(text: str) -> dict:
    """Analyse a marketing text and return a structured risk assessment.

    Parameters
    ----------
    text : str
        The promotional copy to analyse (e.g. product description,
        banner text, or user-pasted marketing content).

    Returns
    -------
    dict
        {
            "risk_score": 0-100,
            "risk_level": "低" | "中" | "高",
            "categories": {
                "时间诱导": {"hits": [...], "sub_score": float},
                "稀缺诱导": {"hits": [...], "sub_score": float},
                "价格诱导": {"hits": [...], "sub_score": float},
            },
            "highlights": [...],   # human-readable flagged phrases
        }
    """
    if not text or not text.strip():
        return _empty_result()

    text_lower = text.lower()

    # --- Scan each category ---
    time_result = _scan_category(text, text_lower, TIME_PRESSURE_KEYWORDS, "时间诱导")
    scarcity_result = _scan_category(text, text_lower, SCARCITY_KEYWORDS, "稀缺诱导")
    price_result = _scan_category(text, text_lower, PRICE_ANCHORING_KEYWORDS, "价格诱导")

    # --- Sub-scores (each 0-100) ---
    time_score = _calc_sub_score(time_result["hits"])
    scarcity_score = _calc_sub_score(scarcity_result["hits"])
    price_score = _calc_sub_score(price_result["hits"])

    # --- Overall score: max of the three categories ---
    # Rationale: a single strong manipulation tactic already raises the flag;
    # we don't want to dilute by averaging.
    overall = round(max(time_score, scarcity_score, price_score), 1)

    # --- Highlights for display ---
    highlights: List[str] = []
    for hit in time_result["hits"]:
        highlights.append(f"[时间压力] 发现 '{hit['keyword']}'")
    for hit in scarcity_result["hits"]:
        highlights.append(f"[稀缺诱导] 发现 '{hit['keyword']}'")

    return {
        "risk_score": overall,
        "risk_level": _score_to_label(overall),
        "categories": {
            "时间诱导": {
                "hits": time_result["hits"],
                "sub_score": round(time_score, 1),
            },
            "稀缺诱导": {
                "hits": scarcity_result["hits"],
                "sub_score": round(scarcity_score, 1),
            },
            "价格诱导": {
                "hits": price_result["hits"],
                "sub_score": round(price_score, 1),
            },
        },
        "highlights": highlights,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _scan_category(
    original: str,
    text_lower: str,
    keywords: List[tuple],
    category: str,
) -> dict:
    """Scan *text_lower* for each keyword in *keywords*, returning hits.

    Searches both the lowercased text (for matching) and the original text
    (for display).  Each keyword only matches once (earliest occurrence).
    """
    hits: List[dict] = []
    seen_positions: set = set()

    for kw, weight in keywords:
        pos = text_lower.find(kw.lower())
        if pos == -1:
            continue
        # Avoid double-counting overlapping hits at exactly the same position
        if pos in seen_positions:
            continue
        seen_positions.add(pos)

        hits.append({
            "keyword": kw,
            "weight": weight,
            "position": pos,
            "context": _extract_context(original, kw, pos),
        })

    # Sort by position for readability
    hits.sort(key=lambda h: h["position"])
    return {"hits": hits, "category": category}


def _extract_context(text: str, keyword: str, pos: int, window: int = 20) -> str:
    """Extract a short window around the keyword for display."""
    start = max(0, pos - window)
    end = min(len(text), pos + len(keyword) + window)
    snippet = text[start:end].replace("\n", " ")
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"
    return snippet


def _calc_sub_score(hits: List[dict]) -> float:
    """Map matched keywords to a 0-100 sub-score.

    Formula: sum(weight) / max_possible * 100, capped at 100.
    max_possible = 15 (roughly 5-6 aggressive keywords in one category).
    """
    if not hits:
        return 0.0
    total_weight = sum(h["weight"] for h in hits)
    MAX_WEIGHT = 15.0
    return min(total_weight / MAX_WEIGHT * 100, 100.0)


def _score_to_label(score: float) -> str:
    if score <= 25:
        return "低"
    elif score <= 55:
        return "中"
    else:
        return "高"


def _empty_result() -> dict:
    return {
        "risk_score": 0.0,
        "risk_level": "低",
        "categories": {
            "时间诱导": {"hits": [], "sub_score": 0.0},
            "稀缺诱导": {"hits": [], "sub_score": 0.0},
            "价格诱导": {"hits": [], "sub_score": 0.0},
        },
        "highlights": [],
    }
