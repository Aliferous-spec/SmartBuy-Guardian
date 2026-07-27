"""
AI purchase advisor — generates natural-language buying advice from a
structured analysis report.

Design principle: the AI is NOT a chatbot. It is a text-transformation engine:
  structured report (dict)  →  LLM  →  structured advice (dict)

The LLM acts as an "analyst" that interprets the numbers and flags and
produces a consumer-friendly summary.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from ai.llm_client import get_client

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
你是一个消费决策分析助手，名为 SmartBuy Guardian。

你的任务是：根据提供的商品价格数据和风险分析结果，给消费者生成购买建议。

规则：
1. 不编造数据中不存在的信息。
2. 不做价格预测（不要说"下周会降价"或"马上要涨价"）。
3. 指出风险点时，要具体说明原因。
4. 用中文输出，语言通俗易懂，200 字以内。
5. 输出严格的 JSON 格式，不要包含其他文字。"""

USER_PROMPT_TEMPLATE = """\
请分析以下商品数据并给出购买建议。

=== 商品信息 ===
名称：{product_name}
当前价格：¥{current_price}
标称原价：{original_price}

=== 价格历史统计 ===
历史最低价：¥{lowest}
历史均价：¥{average}
历史最高价：¥{highest}
价格记录数：{count} 条
近期趋势：{trend_label}

=== 价格诚信分析 ===
风险等级：{risk_level}
风险评分：{risk_score}/100
偏离历史最低：{low_dev}%
偏离历史均价（Z值）：{avg_z}

=== 营销文案分析 ===
营销文本：{marketing_text}
营销风险等级：{mkt_risk_level}
发现的风险话术：{mkt_highlights}

请输出 JSON：
{{
  "verdict": "建议购买 / 建议观望 / 不建议购买",
  "summary": "给消费者的具体建议（200 字以内）",
  "risk_points": ["风险点1", "风险点2"],
  "confidence": 0.0-1.0
}}"""


# ──────────────────────────────────────────────────────────────────────────────
# Fallback advisor (no LLM required)
# ──────────────────────────────────────────────────────────────────────────────


def _fallback_advice(report: dict) -> dict:
    """Generate advice from rules alone, without an LLM call.

    Used when the LLM is not configured or the API call fails.
    """
    price_info = report.get("price_analysis", {})
    mkt_info = report.get("marketing_analysis", {})
    product = report.get("product", {})

    risk_level = price_info.get("risk_level", "未知")
    risk_score = price_info.get("risk_score", 0)
    breakdown = price_info.get("breakdown", {})
    verdict_text = price_info.get("verdict", "")

    # Map risk level to simple verdict
    verdict_map = {
        "低风险": "建议购买",
        "中风险": "建议观望",
        "高风险": "不建议购买",
        "数据不足": "数据不足",
    }
    verdict = verdict_map.get(risk_level, "无法判断")

    # Collect risk points
    risk_points = []
    if breakdown.get("fake_original_price"):
        risk_points.append(breakdown.get("fake_original_detail", "标称原价虚高"))
    low_dev = breakdown.get("deviation_from_low_pct", 0)
    if low_dev > 10:
        risk_points.append(f"当前价格比历史最低价高{low_dev}%")
    mkt_highlights = mkt_info.get("highlights", [])
    for h in mkt_highlights[:3]:
        risk_points.append(h)

    # Simple confidence: higher risk → higher confidence in the "don't buy" signal
    confidence = round(min(risk_score / 100, 1.0), 2)

    return {
        "verdict": verdict,
        "summary": verdict_text,
        "risk_points": risk_points if risk_points else ["未发现明显风险"],
        "confidence": confidence,
        "_source": "rule-engine",  # marks this as non-LLM output
    }


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────


def get_advice(report: dict) -> dict:
    """Generate a consumer-friendly purchase recommendation.

    Parameters
    ----------
    report : dict
        The structured report from :func:`analysis.report_builder.build_report`.

    Returns
    -------
    dict
        {
            "verdict": "建议购买" | "建议观望" | "不建议购买",
            "summary": str,
            "risk_points": [str, ...],
            "confidence": float,
            "_source": "llm" | "rule-engine",
        }
    """
    client = get_client()

    if not client.is_configured:
        logger.info("LLM not configured — using rule-based fallback advice")
        return _fallback_advice(report)

    # ── Build the prompt ──────────────────────────────────────────────────
    product = report.get("product", {})
    price_analysis = report.get("price_analysis", {})
    mkt_analysis = report.get("marketing_analysis", {})
    breakdown = price_analysis.get("breakdown", {})

    original_price = product.get("original_price")
    original_display = f"¥{original_price:.0f}" if original_price else "未标注"

    mkt_text = ""  # we don't store raw text in report currently

    user_prompt = USER_PROMPT_TEMPLATE.format(
        product_name=product.get("name", "未知商品"),
        current_price=f"{product.get('current_price', 0):.2f}",
        original_price=original_display,
        lowest=f"{breakdown.get('historical_lowest', 0):.2f}",
        average=f"{breakdown.get('historical_average', 0):.2f}",
        highest=f"{breakdown.get('historical_highest', 0):.2f}",
        count=breakdown.get("record_count", 0),
        trend_label=breakdown.get("trend_label", "未知"),
        risk_level=price_analysis.get("risk_level", "未知"),
        risk_score=price_analysis.get("risk_score", 0),
        low_dev=breakdown.get("deviation_from_low_pct", 0),
        avg_z=breakdown.get("deviation_from_avg_z", 0),
        marketing_text=mkt_text or "无",
        mkt_risk_level=mkt_analysis.get("risk_level", "低"),
        mkt_highlights="; ".join(mkt_analysis.get("highlights", [])) or "无",
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # ── Call LLM ──────────────────────────────────────────────────────────
    logger.info("Calling LLM for purchase advice...")
    result = client.chat_json(
        messages,
        temperature=0.3,
        max_tokens=800,
        response_format={"type": "json_object"},
    )

    if result is None:
        logger.warning("LLM call failed — falling back to rule-based advice")
        return _fallback_advice(report)

    result["_source"] = "llm"
    return result
