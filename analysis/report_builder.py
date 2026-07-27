"""
Report builder — combines price integrity + marketing risk into a single
structured report that is ready to be displayed or fed to the LLM advisor.
"""

from __future__ import annotations

from typing import Optional


def build_report(
    product_name: str,
    current_price: float,
    original_price: Optional[float],
    history_prices: list[float],
    marketing_text: str,
) -> dict:
    """Run both analysers and return a unified report dict.

    Parameters
    ----------
    product_name : str
        Human-readable product name.
    current_price : float
        Current displayed price.
    original_price : float or None
        "Original" / "list" price shown on page (划线价).
    history_prices : list[float]
        Historical price data-points (oldest → newest).
    marketing_text : str
        Promotional copy from the product page or user input.

    Returns
    -------
    dict
        Structured report ready for display or AI consumption.
    """
    from analysis.marketing_risk import evaluate as marketing_evaluate
    from analysis.price_integrity import assess_price_integrity

    # 1. Marketing risk
    mkt_result = marketing_evaluate(marketing_text)

    # 2. Price integrity (inject marketing risk score)
    price_result = assess_price_integrity(
        history_prices=history_prices,
        current_price=current_price,
        original_price=original_price,
        marketing_risk_score=mkt_result["risk_score"],
    )

    # 3. Merge
    return {
        "product": {
            "name": product_name,
            "current_price": current_price,
            "original_price": original_price,
        },
        "price_analysis": price_result,
        "marketing_analysis": mkt_result,
        "generated_at": None,  # filled by caller if needed
    }
