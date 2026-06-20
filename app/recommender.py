from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.spk import (
    UserPreference,
    age_query,
    age_score,
    family_score,
    fuzzy_saw_score,
    gender_score,
    normalize_gender,
    price_score,
    rating_score,
    usage_query,
)
from app.utils import clean_text, format_price, normalize_for_match, parse_float, parse_int
from app.utils import tokenize_terms


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATASET_PATH = BASE_DIR / "Datasets" / "final_df.csv"
CLEAN_DATASET_PATH = BASE_DIR / "Datasets" / "perfume_clean.csv"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.joblib"
MATRIX_PATH = BASE_DIR / "models" / "tfidf_matrix.joblib"
SBERT_ARTIFACT_DIR = BASE_DIR / "models" / "sbert_fuzzy_topsis"
SBERT_MODEL_PATH = SBERT_ARTIFACT_DIR / "fine_tuned_model"
SBERT_EMBEDDINGS_PATH = SBERT_ARTIFACT_DIR / "perfume_embeddings.npy"
SBERT_DATASET_PATH = BASE_DIR / "reports" / "sbert_fuzzy_topsis" / "prepared_perfume_dataset.csv"
SBERT_BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATASET_PATH = RAW_DATASET_PATH

TOPSIS_WEIGHTS = {
    "semantic_similarity": 0.30,
    "family_fit": 0.16,
    "gender_fit": 0.12,
    "age_fit": 0.08,
    "usage_fit": 0.10,
    "rating_fit": 0.10,
    "popularity_fit": 0.05,
    "budget_fit": 0.09,
}

USAGE_TERMS = {
    "daily": {"fresh", "citrus", "aquatic", "clean", "light", "green"},
    "campus": {"fresh", "citrus", "fruity", "aquatic", "light"},
    "work": {"woody", "floral", "musk", "aromatic", "clean", "elegant"},
    "formal": {"woody", "amber", "musk", "oriental", "leather", "sophisticated"},
    "date": {"sweet", "floral", "musk", "vanilla", "romantic", "warm"},
    "night": {"amber", "oriental", "musk", "vanilla", "woody", "leather"},
}

AGE_TERMS = [
    (0, 20, {"fresh", "citrus", "fruity", "sweet"}),
    (21, 30, {"fresh", "aquatic", "floral", "woody", "sweet"}),
    (31, 45, {"woody", "spicy", "amber", "musk", "aromatic", "oud"}),
    (46, 200, {"oriental", "leather", "amber", "powdery", "musk", "oud"}),
]

RELATED_FAMILY_GROUPS = [
    {"fresh", "citrus", "aquatic", "green", "fruity"},
    {"floral", "soft", "sweet", "fruity", "romantic"},
    {"woody", "aromatic", "spicy", "oud", "leather"},
    {"oriental", "amber", "musk", "arabian", "soft", "oud"},
    {"sweet", "vanilla", "gourmand", "fruity", "caramel"},
]


class PerfumeRecommender:
    def __init__(self, dataset_path: Path | None = None) -> None:
        self.dataset_path = dataset_path or RAW_DATASET_PATH
        self.model_source = "runtime_fit"
        self.embedding_model = None
        self.embeddings: np.ndarray | None = None

        if dataset_path is None and self._sbert_artifacts_available():
            try:
                self.dataset_path = SBERT_DATASET_PATH
                self.df = self._load_sbert_dataset(SBERT_DATASET_PATH)
                self.embeddings = np.load(SBERT_EMBEDDINGS_PATH)
                if self.embeddings.shape[0] != len(self.df):
                    raise ValueError("SBERT embedding row count does not match prepared dataset.")

                from sentence_transformers import SentenceTransformer

                model_path = SBERT_MODEL_PATH if SBERT_MODEL_PATH.exists() else SBERT_BASE_MODEL
                self.embedding_model = SentenceTransformer(str(model_path))
                self.model_source = "sbert_fuzzy_topsis_notebook"
                return
            except Exception:
                self.dataset_path = dataset_path or RAW_DATASET_PATH
                self.embedding_model = None
                self.embeddings = None

        if dataset_path is None and self._tfidf_artifacts_available():
            try:
                self.dataset_path = CLEAN_DATASET_PATH
                self.df = self._load_clean_dataset(CLEAN_DATASET_PATH)
                self.vectorizer = joblib.load(VECTORIZER_PATH)
                self.tfidf_matrix = joblib.load(MATRIX_PATH)
                if self.tfidf_matrix.shape[0] != len(self.df):
                    raise ValueError("TF-IDF matrix row count does not match cleaned dataset.")
                self.model_source = "tfidf_notebook_artifact"
                return
            except Exception:
                self.dataset_path = RAW_DATASET_PATH

        self.df = self._load_dataset(self.dataset_path)
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            strip_accents="unicode",
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["aroma_profile"])

    @staticmethod
    def _sbert_artifacts_available() -> bool:
        return SBERT_DATASET_PATH.exists() and SBERT_EMBEDDINGS_PATH.exists()

    @staticmethod
    def _tfidf_artifacts_available() -> bool:
        return CLEAN_DATASET_PATH.exists() and VECTORIZER_PATH.exists() and MATRIX_PATH.exists()

    def _load_sbert_dataset(self, dataset_path: Path) -> pd.DataFrame:
        df = pd.read_csv(dataset_path)
        if "perfume_id" not in df.columns:
            df = df.reset_index(drop=True)
            df["perfume_id"] = df.index
        df = self._finalize_dataset(df)
        return self._ensure_profiles(df)

    def _load_clean_dataset(self, dataset_path: Path) -> pd.DataFrame:
        df = pd.read_csv(dataset_path)
        if "perfume_id" not in df.columns:
            df = df.reset_index(drop=True)
            df["perfume_id"] = df.index
        return self._ensure_profiles(self._finalize_dataset(df))

    def _load_dataset(self, dataset_path: Path) -> pd.DataFrame:
        df = pd.read_csv(dataset_path)
        df = df.rename(
            columns={
                "Name": "name",
                "Price": "price",
                "Description": "description",
                "Rate": "rating",
                "Rating_count": "rating_count",
                "Brand": "brand",
                "Gender": "gender",
                "Product_Type": "product_type",
                "Character_x": "character",
                "Fragrance_Family": "fragrance_family",
                "Size": "size",
                "Year": "year",
                "Ingredients": "ingredients",
                "Concentration": "concentration",
                "Top_note": "top_note",
                "Middle_note": "middle_note",
                "Base_note": "base_note",
            }
        )

        text_columns = [
            "name",
            "description",
            "image",
            "brand",
            "gender",
            "product_type",
            "character",
            "fragrance_family",
            "size",
            "ingredients",
            "concentration",
            "top_note",
            "middle_note",
            "base_note",
        ]
        for column in text_columns:
            if column not in df.columns:
                df[column] = ""
            df[column] = df[column].map(clean_text)

        df["price"] = pd.to_numeric(df.get("price", 0), errors="coerce").fillna(0.0)
        df["rating"] = pd.to_numeric(df.get("rating", 0), errors="coerce").fillna(0.0)
        df["rating_count"] = pd.to_numeric(df.get("rating_count", 0), errors="coerce").fillna(0).astype(int)
        df["year"] = pd.to_numeric(df.get("year", 0), errors="coerce").fillna(0).astype(int)

        df["gender"] = df["gender"].map(normalize_gender)
        product_mask = df["product_type"].str.contains("perfume", case=False, na=False)
        gender_mask = df["gender"].ne("Home")
        df = df[product_mask & gender_mask].copy()

        df = df.drop_duplicates(subset=["name", "brand", "size"], keep="first")
        df = df.reset_index(drop=True)
        df["perfume_id"] = df.index
        df["fragrance_family"] = df["fragrance_family"].str.replace(r"\s+", " ", regex=True).str.strip()
        df["aroma_profile"] = (
            df["fragrance_family"]
            + " "
            + df["character"]
            + " "
            + df["top_note"]
            + " "
            + df["middle_note"]
            + " "
            + df["base_note"]
            + " "
            + df["ingredients"]
            + " "
            + df["description"]
        ).map(normalize_for_match)
        return self._ensure_profiles(self._finalize_dataset(df))

    def _finalize_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        text_columns = [
            "name",
            "description",
            "image",
            "brand",
            "gender",
            "product_type",
            "character",
            "fragrance_family",
            "size",
            "ingredients",
            "concentration",
            "top_note",
            "middle_note",
            "base_note",
            "aroma_profile",
        ]
        for column in text_columns:
            if column not in df.columns:
                df[column] = ""
            df[column] = df[column].map(clean_text)

        df["price"] = pd.to_numeric(df.get("price", 0), errors="coerce").fillna(0.0)
        df["rating"] = pd.to_numeric(df.get("rating", 0), errors="coerce").fillna(0.0)
        df["rating_count"] = pd.to_numeric(df.get("rating_count", 0), errors="coerce").fillna(0).astype(int)
        default_ids = pd.Series(df.index, index=df.index)
        df["perfume_id"] = pd.to_numeric(df.get("perfume_id", default_ids), errors="coerce").fillna(default_ids).astype(int)
        df["gender"] = df["gender"].map(normalize_gender)
        df["display_price"] = df["price"].map(format_price)
        df["display_rating"] = df["rating"].map(lambda value: f"{value:.2f}".rstrip("0").rstrip(".") if value > 0 else "Belum ada")
        return df

    def _ensure_profiles(self, df: pd.DataFrame) -> pd.DataFrame:
        if "aroma_profile" not in df.columns or df["aroma_profile"].map(clean_text).eq("").all():
            df["aroma_profile"] = (
                df["fragrance_family"]
                + " "
                + df["character"]
                + " "
                + df["top_note"]
                + " "
                + df["middle_note"]
                + " "
                + df["base_note"]
                + " "
                + df["ingredients"]
                + " "
                + df["description"]
            ).map(normalize_for_match)

        if "semantic_text" not in df.columns or df["semantic_text"].map(clean_text).eq("").all():
            df["semantic_text"] = df.apply(self._build_semantic_text, axis=1)
        else:
            df["semantic_text"] = df["semantic_text"].map(clean_text)

        return df

    @staticmethod
    def _build_semantic_text(row: pd.Series) -> str:
        parts = [
            f"Perfume name: {clean_text(row.get('name'))}",
            f"Brand: {clean_text(row.get('brand'))}",
            f"Gender: {clean_text(row.get('gender'))}",
            f"Fragrance family: {clean_text(row.get('fragrance_family'))}",
            f"Character: {clean_text(row.get('character'))}",
            f"Concentration: {clean_text(row.get('concentration'))}",
            f"Top notes: {clean_text(row.get('top_note'))}",
            f"Middle notes: {clean_text(row.get('middle_note'))}",
            f"Base notes: {clean_text(row.get('base_note'))}",
            f"Ingredients: {clean_text(row.get('ingredients'))}",
            f"Description: {clean_text(row.get('description'))}",
        ]
        return ". ".join(part for part in parts if part.split(":", 1)[-1].strip())

    @property
    def total_items(self) -> int:
        return int(len(self.df))

    def fragrance_families(self) -> list[str]:
        families = sorted(
            family
            for family in self.df["fragrance_family"].dropna().unique().tolist()
            if clean_text(family)
        )
        return families

    def recommend(self, preference: UserPreference) -> list[dict[str, Any]]:
        if self.model_source == "sbert_fuzzy_topsis_notebook":
            return self._recommend_sbert_topsis(preference)
        return self._recommend_tfidf_saw(preference)

    def _recommend_tfidf_saw(self, preference: UserPreference) -> list[dict[str, Any]]:
        candidates = self.df.copy()
        if preference.minimal_rating > 0:
            filtered = candidates[candidates["rating"].ge(preference.minimal_rating)].copy()
            if not filtered.empty:
                candidates = filtered

        query = self._build_query(preference)
        query_vector = self.vectorizer.transform([query])
        candidate_matrix = self.tfidf_matrix[candidates.index]
        similarity = cosine_similarity(query_vector, candidate_matrix).flatten()
        candidates = candidates.assign(aroma_similarity=similarity)

        score_rows = []
        for _, row in candidates.iterrows():
            scores = {
                "gender_score": gender_score(preference.gender, row["gender"]),
                "age_score": age_score(preference.age, row["fragrance_family"], row["aroma_profile"]),
                "aroma_similarity": float(row["aroma_similarity"]),
                "rating_score": rating_score(row["rating"]),
                "price_score": price_score(row["price"], preference.budget),
                "family_score": family_score(
                    preference.fragrance_family,
                    preference.aroma_keywords,
                    row["fragrance_family"],
                    row["aroma_profile"],
                ),
            }
            component_scores = {key: value for key, value in scores.items() if key != "aroma_similarity"}
            score_rows.append(component_scores | {"final_score": fuzzy_saw_score(scores)})

        scores_df = pd.DataFrame(score_rows, index=candidates.index)
        ranked = pd.concat([candidates, scores_df], axis=1)
        ranked = ranked.sort_values(
            by=["final_score", "aroma_similarity", "rating", "rating_count"],
            ascending=[False, False, False, False],
        )
        ranked = ranked.head(max(preference.top_n, 1)).copy()
        ranked["rank"] = range(1, len(ranked) + 1)
        ranked["reason"] = ranked.apply(lambda row: self._reason(row, preference), axis=1)

        return ranked[
            [
                "rank",
                "perfume_id",
                "name",
                "brand",
                "gender",
                "price",
                "display_price",
                "rating",
                "display_rating",
                "rating_count",
                "image",
                "fragrance_family",
                "character",
                "top_note",
                "middle_note",
                "base_note",
                "aroma_similarity",
                "gender_score",
                "age_score",
                "rating_score",
                "price_score",
                "family_score",
                "final_score",
                "reason",
            ]
        ].to_dict(orient="records")

    def _recommend_sbert_topsis(self, preference: UserPreference) -> list[dict[str, Any]]:
        if self.embedding_model is None or self.embeddings is None:
            return self._recommend_tfidf_saw(preference)

        candidates = self.df.copy()
        if preference.minimal_rating > 0:
            filtered = candidates[candidates["rating"].ge(preference.minimal_rating)].copy()
            if not filtered.empty:
                candidates = filtered

        query = self._build_sbert_query(preference)
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        similarity = self.embeddings @ query_embedding
        candidates = candidates.assign(aroma_similarity=similarity[candidates.index])

        score_rows = []
        for _, row in candidates.iterrows():
            note_terms = tokenize_terms(
                row["fragrance_family"],
                row["character"],
                row["top_note"],
                row["middle_note"],
                row["base_note"],
                row["ingredients"],
                row["description"],
            )
            score_rows.append(
                {
                    "semantic_similarity": float(np.clip((row["aroma_similarity"] + 1.0) / 2.0, 0.0, 1.0)),
                    "family_fit": self._family_fit(preference, row, note_terms),
                    "gender_fit": gender_score(preference.gender, row["gender"]),
                    "age_fit": self._age_fit(preference.age, note_terms),
                    "usage_fit": self._usage_fit(preference.usage, note_terms),
                    "rating_fit": self._rating_fit(row["rating"], row["rating_count"]),
                    "popularity_fit": self._popularity_fit(row["rating_count"]),
                    "budget_fit": self._budget_fit(row["price"], preference.budget),
                }
            )

        scores_df = pd.DataFrame(score_rows, index=candidates.index)
        topsis_df = self._fuzzy_topsis(scores_df, TOPSIS_WEIGHTS)
        ranked = pd.concat([candidates, topsis_df], axis=1)
        ranked["final_score"] = ranked["topsis_score"]

        ranked["gender_score"] = ranked["gender_fit"]
        ranked["age_score"] = ranked["age_fit"]
        ranked["rating_score"] = ranked["rating_fit"]
        ranked["price_score"] = ranked["budget_fit"]
        ranked["family_score"] = ranked["family_fit"]

        ranked = ranked.sort_values(
            by=["final_score", "aroma_similarity", "rating", "rating_count"],
            ascending=[False, False, False, False],
        )
        ranked = ranked.head(max(preference.top_n, 1)).copy()
        ranked["rank"] = range(1, len(ranked) + 1)
        ranked["reason"] = ranked.apply(lambda row: self._reason(row, preference), axis=1)

        return ranked[
            [
                "rank",
                "perfume_id",
                "name",
                "brand",
                "gender",
                "price",
                "display_price",
                "rating",
                "display_rating",
                "rating_count",
                "image",
                "fragrance_family",
                "character",
                "top_note",
                "middle_note",
                "base_note",
                "aroma_similarity",
                "semantic_similarity",
                "gender_score",
                "age_score",
                "usage_fit",
                "rating_score",
                "price_score",
                "family_score",
                "popularity_fit",
                "final_score",
                "reason",
            ]
        ].to_dict(orient="records")

    def _build_sbert_query(self, preference: UserPreference) -> str:
        usage_key = normalize_for_match(preference.usage)
        usage_terms = " ".join(sorted(USAGE_TERMS.get(usage_key, set())))
        parts = [
            preference.fragrance_family,
            preference.aroma_keywords,
            usage_terms,
            age_query(preference.age),
            preference.gender,
        ]
        query = " ".join(part for part in parts if clean_text(part))
        if not query.strip():
            return "fresh floral woody citrus musk"
        return query

    def _family_fit(self, preference: UserPreference, row: pd.Series, note_terms: set[str]) -> float:
        preferred_family = normalize_for_match(preference.fragrance_family)
        row_family = normalize_for_match(row["fragrance_family"])
        preferred_terms = tokenize_terms(preference.fragrance_family, preference.aroma_keywords)

        if not preferred_family and not preferred_terms:
            return 0.65
        if preferred_family and preferred_family == row_family:
            return 1.0
        if preferred_family and (preferred_family in row_family or row_family in preferred_family):
            return 0.9
        if preferred_terms and preferred_terms & note_terms:
            return 0.8
        if self._related_family_score(preference.fragrance_family, row["fragrance_family"]) >= 0.65:
            return 0.7
        return 0.25

    @staticmethod
    def _age_fit(age: int, note_terms: set[str]) -> float:
        target_terms = set()
        for lower, upper, terms in AGE_TERMS:
            if lower <= age <= upper:
                target_terms = terms
                break
        if not target_terms:
            target_terms = AGE_TERMS[-1][2]

        overlap = len(target_terms & note_terms)
        if overlap >= 2:
            return 1.0
        if overlap == 1:
            return 0.7
        return 0.35

    @staticmethod
    def _usage_fit(usage: str, note_terms: set[str]) -> float:
        target_terms = USAGE_TERMS.get(normalize_for_match(usage), set())
        if not target_terms:
            return 0.65
        overlap = len(target_terms & note_terms)
        if overlap >= 3:
            return 1.0
        if overlap == 2:
            return 0.8
        if overlap == 1:
            return 0.55
        return 0.25

    @staticmethod
    def _rating_fit(rating: Any, rating_count: Any) -> float:
        rating_value = min(max(parse_float(rating, 0.0) / 5.0, 0.0), 1.0)
        count_value = max(parse_int(rating_count, 0), 0)
        confidence = min(np.log1p(count_value) / np.log1p(20), 1.0)
        return float(np.clip((0.75 * rating_value) + (0.25 * rating_value * confidence), 0.0, 1.0))

    def _popularity_fit(self, rating_count: Any) -> float:
        max_count = max(int(self.df["rating_count"].max()), 1)
        count_value = max(parse_int(rating_count, 0), 0)
        return float(np.clip(np.log1p(count_value) / np.log1p(max_count), 0.0, 1.0))

    @staticmethod
    def _budget_fit(price: Any, budget: float) -> float:
        price_value = parse_float(price, 0.0)
        if budget <= 0:
            return 0.70
        if price_value <= budget:
            return 1.0
        if price_value <= budget * 1.15:
            return 0.75
        if price_value <= budget * 1.35:
            return 0.45
        return 0.15

    @staticmethod
    def _related_family_score(left_family: str, right_family: str) -> float:
        left = normalize_for_match(left_family)
        right = normalize_for_match(right_family)
        if not left or not right:
            return 0.3
        if left == right:
            return 1.0

        left_terms = tokenize_terms(left)
        right_terms = tokenize_terms(right)
        if left_terms & right_terms:
            return 0.8

        for group in RELATED_FAMILY_GROUPS:
            if (left_terms & group) and (right_terms & group):
                return 0.65
        return 0.15

    @staticmethod
    def _fuzzy_topsis(decision_matrix: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
        criteria = list(weights.keys())
        normalized_weights = np.array([max(float(weights[criterion]), 0.0) for criterion in criteria])
        total_weight = normalized_weights.sum()
        if total_weight <= 0:
            normalized_weights = np.ones(len(criteria)) / len(criteria)
        else:
            normalized_weights = normalized_weights / total_weight

        matrix = decision_matrix[criteria].astype(float).clip(0.0, 1.0).to_numpy()
        norm = np.sqrt((matrix**2).sum(axis=0))
        norm[norm == 0] = 1.0
        weighted = (matrix / norm) * normalized_weights

        ideal_positive = weighted.max(axis=0)
        ideal_negative = weighted.min(axis=0)
        distance_positive = np.sqrt(((weighted - ideal_positive) ** 2).sum(axis=1))
        distance_negative = np.sqrt(((weighted - ideal_negative) ** 2).sum(axis=1))
        topsis_score = distance_negative / (distance_positive + distance_negative + 1e-12)

        result = decision_matrix.copy()
        result["distance_positive"] = distance_positive
        result["distance_negative"] = distance_negative
        result["topsis_score"] = topsis_score
        return result

    def get_perfume(self, perfume_id: int) -> dict[str, Any] | None:
        match = self.df[self.df["perfume_id"].eq(perfume_id)]
        if match.empty:
            return None
        row = match.iloc[0].to_dict()
        return row

    def _build_query(self, preference: UserPreference) -> str:
        parts = [
            preference.fragrance_family,
            preference.aroma_keywords,
            usage_query(preference.usage),
            age_query(preference.age),
        ]
        query = " ".join(part for part in parts if clean_text(part))
        if not query.strip():
            return "fresh floral woody citrus musk"
        return normalize_for_match(query)

    def _reason(self, row: pd.Series, preference: UserPreference) -> str:
        reasons: list[str] = []
        if row["gender_score"] >= 0.8:
            reasons.append(f"gender {row['gender']} sesuai")
        if row["age_score"] >= 0.9:
            reasons.append("profil aroma cocok dengan kelompok umur")
        if row["family_score"] >= 0.8:
            reasons.append(f"family {row['fragrance_family']} relevan")
        if row["aroma_similarity"] >= 0.25:
            reasons.append("notes aromanya mirip dengan input")
        if row["rating_score"] >= 0.8:
            reasons.append("rating tinggi")
        if row["price_score"] >= 0.9 and preference.budget > 0:
            reasons.append("harga masuk budget")

        if not reasons:
            reasons.append("skor gabungan ML dan SPK paling kompetitif")
        return ", ".join(reasons).capitalize() + "."


def preference_from_mapping(data: Any) -> UserPreference:
    return UserPreference(
        gender=normalize_gender(data.get("gender", "Unisex")),
        age=max(parse_int(data.get("age"), 25), 1),
        fragrance_family=clean_text(data.get("fragrance_family")),
        aroma_keywords=clean_text(data.get("aroma_keywords")),
        budget=max(parse_float(data.get("budget"), 0.0), 0.0),
        minimal_rating=min(max(parse_float(data.get("minimal_rating"), 0.0), 0.0), 5.0),
        usage=clean_text(data.get("usage", "daily")) or "daily",
        top_n=min(max(parse_int(data.get("top_n"), 10), 1), 30),
    )


@lru_cache(maxsize=1)
def get_recommender() -> PerfumeRecommender:
    return PerfumeRecommender()
