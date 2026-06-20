from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.utils import normalize_for_match, parse_float, tokenize_terms


WEIGHTS = {
    "gender_score": 0.20,
    "age_score": 0.15,
    "aroma_similarity": 0.25,
    "rating_score": 0.15,
    "price_score": 0.10,
    "family_score": 0.15,
}

AGE_RULES = [
    (0, 20, {"fresh", "citrus", "fruity", "sweet"}),
    (21, 30, {"fresh", "aquatic", "floral", "woody", "sweet"}),
    (31, 45, {"woody", "spicy", "amber", "musk", "aromatic", "oud"}),
    (46, 200, {"oriental", "leather", "amber", "powdery", "musk", "oud"}),
]

RELATED_FAMILIES = [
    {"fresh", "citrus", "aquatic", "fruity", "green"},
    {"floral", "soft", "sweet", "fruity", "romantic"},
    {"woody", "aromatic", "spicy", "oud", "leather"},
    {"oriental", "amber", "musk", "arabian", "soft", "oud"},
    {"sweet", "vanilla", "gourmand", "fruity", "caramel"},
]

USAGE_KEYWORDS = {
    "daily": "fresh citrus aquatic clean light",
    "campus": "fresh citrus fruity aquatic light",
    "work": "woody floral musk aromatic clean elegant",
    "formal": "woody amber musk oriental sophisticated",
    "date": "sweet floral musk vanilla romantic warm",
}


@dataclass(frozen=True)
class UserPreference:
    gender: str
    age: int
    fragrance_family: str
    aroma_keywords: str
    budget: float
    minimal_rating: float
    usage: str
    top_n: int = 10


def normalize_gender(value: Any) -> str:
    text = normalize_for_match(value)
    if text in {"men", "man", "male", "pria", "laki laki"}:
        return "Men"
    if text in {"women", "woman", "female", "wanita", "perempuan"}:
        return "Women"
    if text in {"unisex", "all", "semua"}:
        return "Unisex"
    return str(value or "").strip().title()


def gender_score(user_gender: str, perfume_gender: str) -> float:
    user = normalize_gender(user_gender)
    perfume = normalize_gender(perfume_gender)

    if not user:
        return 0.7
    if user == perfume:
        return 1.0
    if perfume == "Unisex":
        return 0.8
    if user == "Unisex" and perfume in {"Men", "Women"}:
        return 0.7
    return 0.3


def _age_terms(age: int) -> set[str]:
    for minimum, maximum, terms in AGE_RULES:
        if minimum <= age <= maximum:
            return terms
    return AGE_RULES[-1][2]


def age_group_label(age: int) -> str:
    if age <= 20:
        return "15-20"
    if age <= 30:
        return "21-30"
    if age <= 45:
        return "31-45"
    return "46+"


def age_score(age: int, fragrance_family: str, aroma_text: str) -> float:
    terms = _age_terms(age)
    perfume_terms = tokenize_terms(fragrance_family, aroma_text)

    if terms & perfume_terms:
        return 1.0
    if _has_related_overlap(terms, perfume_terms):
        return 0.7
    return 0.4


def rating_score(rating: Any) -> float:
    value = parse_float(rating, 0.0)
    if value <= 0:
        return 0.0
    return min(value / 5.0, 1.0)


def price_score(price: Any, budget: float) -> float:
    value = parse_float(price, 0.0)
    if budget <= 0:
        return 0.7
    if value <= budget:
        return 1.0
    if value <= budget * 1.2:
        return 0.7
    return 0.3


def family_score(preferred_family: str, aroma_keywords: str, row_family: str, aroma_text: str) -> float:
    preferred = normalize_for_match(preferred_family)
    row = normalize_for_match(row_family)
    perfume_terms = tokenize_terms(row_family, aroma_text)
    preferred_terms = tokenize_terms(preferred_family, aroma_keywords)

    if preferred and (preferred == row or preferred in row or row in preferred):
        return 1.0
    if preferred_terms and preferred_terms & perfume_terms:
        return 0.85
    if preferred_terms and _has_related_overlap(preferred_terms, perfume_terms):
        return 0.7
    if preferred:
        return 0.3
    return 0.6


def fuzzy_saw_score(scores: dict[str, float]) -> float:
    total = 0.0
    for key, weight in WEIGHTS.items():
        total += scores.get(key, 0.0) * weight
    return round(total, 4)


def usage_query(usage: str) -> str:
    key = normalize_for_match(usage)
    return USAGE_KEYWORDS.get(key, "")


def age_query(age: int) -> str:
    return " ".join(sorted(_age_terms(age)))


def _has_related_overlap(left: set[str], right: set[str]) -> bool:
    for group in RELATED_FAMILIES:
        if (left & group) and (right & group):
            return True
    return False
