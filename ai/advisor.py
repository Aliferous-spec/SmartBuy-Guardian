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
你是一个专业、值得信赖的消费决策顾问，名为 SmartBuy Guardian。

你的任务是帮助消费者判断商品的促销价格是否值得购买。

规则：
1. 不编造数据中不存在的信息。
2. 不做价格预测（不要说"下周会降价"或"马上要涨价"）。
3. 语气客观中立，不要使用夸张的营销式语言，要体现"帮消费者说话"而非"帮商家说话"的立场。
4. 用中文输出，语言通俗易懂。
5. 输出严格的 JSON 格式，不要包含其他文字。"""

USER_PROMPT_TEMPLATE = """\
请基于以下数据生成购买建议：

商品名称：{product_name}
当前价格：¥{current_price}
标称原价：{original_price}
历史最低价：¥{lowest}
历史均价：¥{average}
历史最高价：¥{highest}
偏离历史最低价百分比：{low_dev}%
价格记录数：{count} 条
近期趋势：{trend_label}

营销话术风险检测：
{marketing_risk_flags}

请输出 JSON：
{{
  "verdict": "建议购买 / 建议观望 / 不建议购买",
  "summary": "给消费者的具体建议",
  "risk_points": ["风险点1", "风险点2"],
  "confidence": 0.0-1.0
}}

summary 字段要求：
1. 控制在80字以内
2. 明确给出结论（建议购买/建议观望/建议等待）放在开头
3. 用一句话说明判断依据（价格对比和营销话术风险）
4. 如果检测到营销话术风险，明确指出具体套路类型（如"稀缺诱导""限时压力""价格锚定"），并简要说明为什么需要警惕
5. 语气客观中立，体现"帮消费者说话"的立场，不要使用夸张的营销式语言"""


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


def _build_marketing_flags(mkt_analysis: dict) -> str:
    """Build a human-readable summary of marketing risk categories detected.

    Returns a string describing which manipulation tactics were found,
    with concrete examples of the flagged phrases.
    """
    categories = mkt_analysis.get("categories", {})
    if not categories:
        return "未检测到营销话术风险"

    lines = []
    for cat_name, cat_data in categories.items():
        hits = cat_data.get("hits", [])
        if hits:
            # Extract keyword string from each hit (hit may be dict or str)
            keywords = []
            for h in hits[:3]:
                if isinstance(h, dict):
                    keywords.append(h.get("keyword", str(h)))
                else:
                    keywords.append(str(h))
            examples = "、".join(f'"{kw}"' for kw in keywords)
            lines.append(f"- {cat_name}：{examples}")

    if not lines:
        return "未检测到营销话术风险"

    return "\n".join(lines)


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

    marketing_risk_flags = _build_marketing_flags(mkt_analysis)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        product_name=product.get("name", "未知商品"),
        current_price=f"{product.get('current_price', 0):.2f}",
        original_price=original_display,
        lowest=f"{breakdown.get('historical_lowest', 0):.2f}",
        average=f"{breakdown.get('historical_average', 0):.2f}",
        highest=f"{breakdown.get('historical_highest', 0):.2f}",
        low_dev=breakdown.get("deviation_from_low_pct", 0),
        count=breakdown.get("record_count", 0),
        trend_label=breakdown.get("trend_label", "未知"),
        marketing_risk_flags=marketing_risk_flags,
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
