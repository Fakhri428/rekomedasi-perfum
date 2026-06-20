from __future__ import annotations

import re
from typing import Any


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return text


def normalize_for_match(value: Any) -> str:
    text = clean_text(value).lower()
    text = re.sub(r"[^a-z0-9,\s&-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return default


def parse_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(float(str(value).replace(",", ".")))
    except (TypeError, ValueError):
        return default


def format_price(value: Any) -> str:
    price = parse_float(value, 0.0)
    if price >= 1000:
        return f"{price:,.0f}"
    return f"{price:,.2f}".rstrip("0").rstrip(".")


def tokenize_terms(*values: Any) -> set[str]:
    terms: set[str] = set()
    for value in values:
        text = normalize_for_match(value)
        if not text:
            continue
        parts = re.split(r"[,/&\-\s]+", text)
        terms.update(part for part in parts if len(part) > 2)
    return terms
