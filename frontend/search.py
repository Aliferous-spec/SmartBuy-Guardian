# -*- coding: utf-8 -*-
"""
Search module — matches user input against known products in demo_data.py
and loads price history from demo_history.jsonl.

Design:
  - demo_data.py  → product metadata (product_name, original_price, marketing_text)
  - demo_history.jsonl → historical prices (timestamp, price)

Flow:
  1. Extract keywords from user input (handles both plain text and URLs)
  2. Match against demo_data.py product names using simple keyword matching
  3. If matched: merge metadata + price history → analysis-ready case dict
  4. If not matched: return None (caller shows friendly message)
"""

from __future__ import annotations

import json
import os
import re
from typing import List, Optional, Dict, Any

from frontend.demo_data import DEMO_CASES

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_HISTORY_PATH = os.path.join(_PROJECT_ROOT, "demo_history.jsonl")


# ══════════════════════════════════════════════════════════════════════════════
# Price history loading
# ══════════════════════════════════════════════════════════════════════════════

def load_price_history() -> List[float]:
    """Load all price records from demo_history.jsonl and return price list.

    Returns an empty list if the file is missing, empty, or malformed.
    """
    prices: List[float] = []
    if not os.path.exists(_HISTORY_PATH):
        return prices
    with open(_HISTORY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                prices.append(float(record["price"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
    return prices


# ══════════════════════════════════════════════════════════════════════════════
# Keyword extraction
# ══════════════════════════════════════════════════════════════════════════════

def _extract_keywords(query: str) -> List[str]:
    """Extract meaningful keywords from user input.

    Handles both plain text and URLs:
      - URLs: strip protocol + domain, extract path segments as keywords
      - Plain text: split by whitespace, filter short fragments

    Returns keywords of length >= 2 only (single characters are too noisy
    for Chinese matching).
    """
    query = query.strip()
    if not query:
        return []

    # If it looks like a URL, extract path components as keywords
    if query.startswith("http://") or query.startswith("https://"):
        # Remove protocol and domain, keep only the path
        path = re.sub(r'^https?://[^/]+/', '', query)
        # Remove file extension and query string
        path = re.sub(r'\.[^.]+$', '', path)
        path = re.sub(r'\?.*$', '', path)
        # Split by common URL separators
        words = re.split(r'[-/_]', path)
        return [w for w in words if len(w) >= 2]

    # Plain text: split by whitespace (handles Chinese + spaces)
    return [w for w in query.split() if len(w) >= 2]


# ══════════════════════════════════════════════════════════════════════════════
# Product matching
# ══════════════════════════════════════════════════════════════════════════════

def match_product(query: str) -> Optional[Dict[str, Any]]:
    """Simple keyword matching against demo_data.py product names.

    For each keyword extracted from the query, checks whether it appears
    in any demo case's product_name or name field (case-insensitive).

    Returns the first matching DEMO_CASES dict, or None if no match.
    """
    keywords = _extract_keywords(query)
    if not keywords:
        return None

    for case in DEMO_CASES:
        product_name = case.get("product_name", "")
        case_name = case.get("name", "")
        # Combine both fields for matching
        combined = (product_name + " " + case_name).lower()

        for kw in keywords:
            if kw.lower() in combined:
                return case

    return None


# ══════════════════════════════════════════════════════════════════════════════
# Case builder
# ══════════════════════════════════════════════════════════════════════════════

def build_search_case(query: str) -> Optional[Dict[str, Any]]:
    """Build a complete analysis case from user search input.

    Merges:
      - Product metadata from demo_data.py (product_name, original_price, marketing_text)
      - Price history from demo_history.jsonl (list of floats)
      - current_price = last price in history

    Returns a dict compatible with _run_analysis() and _render_result(),
    or None if:
      - No matching product found (caller shows "not found" message)
      - No price history available (caller shows "not found" message)
    """
    matched = match_product(query)
    if matched is None:
        return None

    prices = load_price_history()
    if not prices:
        return None

    # Merge metadata from demo_data.py with price history from demo_history.jsonl
    return {
        "name": matched.get("name", matched.get("product_name", "")),
        "product_name": matched.get("product_name", ""),
        "current_price": prices[-1],                      # latest recorded price
        "original_price": matched.get("original_price"),  # None if missing → analysis handles it
        "marketing_text": matched.get("marketing_text", ""),  # "" if missing → skip marketing analysis
        "history_prices": prices,
    }
