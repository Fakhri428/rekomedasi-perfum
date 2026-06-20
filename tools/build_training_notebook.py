from pathlib import Path
from textwrap import dedent

import nbformat as nbf


def md(source: str):
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(dedent(source).strip())


nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "pygments_lexer": "ipython3",
    },
}

nb.cells = [
    md(
        """
        # Training dan Evaluasi Sistem Rekomendasi Parfum

        Notebook ini adalah tahap eksperimen utama sebelum aplikasi Flask dibuat.
        Alurnya:

        1. Load `Datasets/final_df.csv`.
        2. Data understanding dan preprocessing.
        3. Training model TF-IDF untuk fitur aroma parfum.
        4. Evaluasi model similarity menggunakan metrik retrieval Top-K.
        5. Implementasi rule dan Fuzzy SAW.
        6. Uji skenario rekomendasi.
        7. Simpan dataset bersih, model, matrix, dan metrik evaluasi untuk dipakai Flask.

        Catatan metodologi: project ini adalah hybrid recommender, bukan klasifikasi supervised.
        Karena dataset tidak memiliki label "cocok/tidak cocok", evaluasi model memakai proxy label
        `fragrance_family` dan metrik retrieval seperti Precision@K, Recall@K, HitRate@K, MRR@K,
        dan nDCG@K.
        """
    ),
    md("## 1. Import Library dan Konfigurasi Path"),
    code(
        """
        from pathlib import Path
        import json
        import re

        import joblib
        import numpy as np
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.model_selection import train_test_split

        pd.set_option("display.max_columns", 100)
        pd.set_option("display.max_colwidth", 120)

        PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
        DATASET_PATH = PROJECT_ROOT / "Datasets" / "final_df.csv"
        PROCESSED_PATH = PROJECT_ROOT / "Datasets" / "perfume_clean.csv"
        MODEL_DIR = PROJECT_ROOT / "models"
        REPORT_DIR = PROJECT_ROOT / "reports"

        MODEL_DIR.mkdir(exist_ok=True)
        REPORT_DIR.mkdir(exist_ok=True)

        VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
        MATRIX_PATH = MODEL_DIR / "tfidf_matrix.joblib"
        METADATA_PATH = MODEL_DIR / "metadata.json"
        EVALUATION_PATH = REPORT_DIR / "model_evaluation.json"

        DATASET_PATH
        """
    ),
    md("## 2. Load Dataset Mentah"),
    code(
        """
        df_raw = pd.read_csv(DATASET_PATH)
        df_raw.shape
        """
    ),
    code("df_raw.head()"),
    code(
        """
        pd.DataFrame({
            "dtype": df_raw.dtypes.astype(str),
            "missing": df_raw.isna().sum(),
            "unique": df_raw.nunique(dropna=True),
        })
        """
    ),
    md(
        """
        ## 3. Data Understanding Ringkas

        Kolom yang penting untuk model:

        - `Name`, `Brand`, `Gender`, `Product_Type`
        - `Price`, `Rate`, `Rating_count`
        - `Fragrance_Family`, `Character_x`
        - `Top_note`, `Middle_note`, `Base_note`, `Ingredients`, `Description`
        """
    ),
    code(
        """
        display(df_raw["Product_Type"].fillna("NA").value_counts().head(15))
        display(df_raw["Gender"].fillna("NA").value_counts())
        display(df_raw["Fragrance_Family"].fillna("NA").value_counts().head(20))
        display(df_raw["Rate"].fillna("NA").value_counts().head(15))
        """
    ),
    md("## 4. Fungsi Preprocessing"),
    code(
        r"""
        def clean_text(value):
            if value is None:
                return ""
            text = str(value).strip()
            if text.lower() in {"nan", "none", "null"}:
                return ""
            return text


        def normalize_text(value):
            text = clean_text(value).lower()
            text = re.sub(r"[^a-z0-9,\s&-]", " ", text)
            return re.sub(r"\s+", " ", text).strip()


        def parse_float(value, default=0.0):
            try:
                if value is None or str(value).strip() == "":
                    return default
                return float(str(value).replace(",", "."))
            except (TypeError, ValueError):
                return default


        def normalize_gender(value):
            text = normalize_text(value)
            if text in {"men", "man", "male", "pria", "laki laki"}:
                return "Men"
            if text in {"women", "woman", "female", "wanita", "perempuan"}:
                return "Women"
            if text in {"unisex", "all", "semua"}:
                return "Unisex"
            return clean_text(value).title()


        def tokenize_terms(*values):
            terms = set()
            for value in values:
                text = normalize_text(value)
                if not text:
                    continue
                parts = re.split(r"[,/&\-\s]+", text)
                terms.update(part for part in parts if len(part) > 2)
            return terms
        """
    ),
    code(
        r"""
        def preprocess_perfume_data(raw_df):
            df = raw_df.rename(columns={
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
            }).copy()

            text_columns = [
                "name", "description", "image", "brand", "gender", "product_type",
                "character", "fragrance_family", "size", "ingredients",
                "concentration", "top_note", "middle_note", "base_note",
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
                df["fragrance_family"] + " " +
                df["character"] + " " +
                df["top_note"] + " " +
                df["middle_note"] + " " +
                df["base_note"] + " " +
                df["ingredients"] + " " +
                df["description"]
            ).map(normalize_text)

            df["display_price"] = df["price"].map(lambda value: f"{value:,.2f}".rstrip("0").rstrip("."))
            df["display_rating"] = df["rating"].map(
                lambda value: f"{value:.2f}".rstrip("0").rstrip(".") if value > 0 else "Belum ada"
            )
            return df


        df = preprocess_perfume_data(df_raw)
        df.shape
        """
    ),
    code(
        """
        df[[
            "perfume_id", "name", "brand", "gender", "product_type", "price",
            "rating", "fragrance_family", "aroma_profile"
        ]].head()
        """
    ),
    md("## 5. EDA Setelah Cleaning"),
    code(
        """
        display(df["gender"].value_counts())
        display(df["fragrance_family"].value_counts().head(20))
        display(df[["price", "rating", "rating_count"]].describe())
        """
    ),
    md(
        """
        ## 6. Training Model TF-IDF

        Pada model TF-IDF, proses training adalah `fit` terhadap corpus teks.
        Corpus yang digunakan adalah `aroma_profile`, yaitu gabungan fragrance family,
        character, top notes, middle notes, base notes, ingredients, dan description.

        Untuk evaluasi offline, data dibagi menjadi train/test. Setelah evaluasi selesai,
        vectorizer final akan di-fit ulang pada seluruh dataset bersih dan disimpan untuk Flask.
        """
    ),
    code(
        """
        evaluable_df = df[df["fragrance_family"].ne("")].copy()
        family_counts = evaluable_df["fragrance_family"].value_counts()
        evaluable_df = evaluable_df[evaluable_df["fragrance_family"].isin(family_counts[family_counts >= 3].index)].copy()

        train_df, test_df = train_test_split(
            evaluable_df,
            test_size=0.2,
            random_state=42,
            stratify=evaluable_df["fragrance_family"],
        )

        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        tfidf_eval = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            strip_accents="unicode",
        )
        train_matrix = tfidf_eval.fit_transform(train_df["aroma_profile"])

        train_df.shape, test_df.shape, train_matrix.shape
        """
    ),
    md(
        """
        ## 7. Evaluasi Model Similarity

        Karena tidak ada label kecocokan user-parfum, evaluasi memakai proxy:

        - Query test dibuat dari `character`, `top_note`, `middle_note`, `base_note`, dan `ingredients`.
        - Item dianggap relevan jika memiliki `fragrance_family` yang sama.
        - Hasil dibandingkan dengan random baseline.

        Metrik:

        - Precision@K: proporsi hasil Top-K yang relevan.
        - Recall@K: proporsi item relevan yang berhasil ditemukan.
        - HitRate@K: minimal satu item relevan muncul di Top-K.
        - MRR@K: posisi relevan pertama.
        - nDCG@K: kualitas urutan ranking.
        """
    ),
    code(
        """
        def dcg_at_k(relevance):
            relevance = np.asarray(relevance, dtype=float)
            if relevance.size == 0:
                return 0.0
            discounts = np.log2(np.arange(2, relevance.size + 2))
            return float(np.sum(relevance / discounts))


        def metrics_for_indices(retrieved_indices, relevant_mask, k):
            top_indices = np.asarray(retrieved_indices[:k])
            rel = relevant_mask[top_indices].astype(int)
            hits = int(rel.sum())
            total_relevant = int(relevant_mask.sum())
            precision = hits / k
            recall = hits / total_relevant if total_relevant else 0.0
            hit_rate = 1.0 if hits > 0 else 0.0

            relevant_positions = np.where(rel == 1)[0]
            mrr = 1.0 / (relevant_positions[0] + 1) if len(relevant_positions) else 0.0

            ideal_relevance = np.ones(min(total_relevant, k))
            ndcg = dcg_at_k(rel) / dcg_at_k(ideal_relevance) if total_relevant else 0.0
            return precision, recall, hit_rate, mrr, ndcg


        def evaluate_similarity_model(train_df, test_df, vectorizer, train_matrix, k_values=(5, 10), max_queries=500, seed=42):
            rng = np.random.default_rng(seed)
            sample_df = test_df.sample(n=min(max_queries, len(test_df)), random_state=seed).reset_index(drop=True)
            rows = []

            for _, row in sample_df.iterrows():
                query = normalize_text(
                    " ".join([
                        row["character"],
                        row["top_note"],
                        row["middle_note"],
                        row["base_note"],
                        row["ingredients"],
                    ])
                )
                if not query:
                    continue

                query_vector = vectorizer.transform([query])
                scores = cosine_similarity(query_vector, train_matrix).flatten()
                ranked_indices = np.argsort(scores)[::-1]
                relevant_mask = train_df["fragrance_family"].eq(row["fragrance_family"]).to_numpy()

                random_indices = rng.permutation(len(train_df))

                for k in k_values:
                    precision, recall, hit_rate, mrr, ndcg = metrics_for_indices(ranked_indices, relevant_mask, k)
                    rows.append({
                        "model": "tfidf_cosine",
                        "k": k,
                        "precision": precision,
                        "recall": recall,
                        "hit_rate": hit_rate,
                        "mrr": mrr,
                        "ndcg": ndcg,
                    })

                    precision, recall, hit_rate, mrr, ndcg = metrics_for_indices(random_indices, relevant_mask, k)
                    rows.append({
                        "model": "random_baseline",
                        "k": k,
                        "precision": precision,
                        "recall": recall,
                        "hit_rate": hit_rate,
                        "mrr": mrr,
                        "ndcg": ndcg,
                    })

            return pd.DataFrame(rows)


        evaluation_rows = evaluate_similarity_model(train_df, test_df, tfidf_eval, train_matrix)
        evaluation_summary = (
            evaluation_rows
            .groupby(["model", "k"], as_index=False)
            .mean(numeric_only=True)
            .sort_values(["k", "model"])
        )
        evaluation_summary
        """
    ),
    code(
        """
        evaluation_pivot = evaluation_summary.pivot(index="k", columns="model", values=["precision", "recall", "hit_rate", "mrr", "ndcg"])
        evaluation_pivot
        """
    ),
    md("## 8. Training Final dan Simpan Artifact Model"),
    code(
        """
        final_vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            strip_accents="unicode",
        )
        final_tfidf_matrix = final_vectorizer.fit_transform(df["aroma_profile"])

        df.to_csv(PROCESSED_PATH, index=False)
        joblib.dump(final_vectorizer, VECTORIZER_PATH)
        joblib.dump(final_tfidf_matrix, MATRIX_PATH)

        metadata = {
            "dataset_source": str(DATASET_PATH.relative_to(PROJECT_ROOT)),
            "processed_dataset": str(PROCESSED_PATH.relative_to(PROJECT_ROOT)),
            "vectorizer": str(VECTORIZER_PATH.relative_to(PROJECT_ROOT)),
            "tfidf_matrix": str(MATRIX_PATH.relative_to(PROJECT_ROOT)),
            "n_items": int(len(df)),
            "n_features": int(final_tfidf_matrix.shape[1]),
            "training_method": "TfidfVectorizer.fit_transform(aroma_profile)",
            "evaluation_method": "Top-K retrieval with fragrance_family as proxy relevance label",
        }
        METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        metadata
        """
    ),
    md("## 9. Rule SPK dan Fuzzy SAW"),
    code(
        """
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


        def has_related_overlap(left, right):
            for group in RELATED_FAMILIES:
                if (left & group) and (right & group):
                    return True
            return False


        def get_age_terms(age):
            for minimum, maximum, terms in AGE_RULES:
                if minimum <= age <= maximum:
                    return terms
            return AGE_RULES[-1][2]


        def gender_score(user_gender, perfume_gender):
            user = normalize_gender(user_gender)
            perfume = normalize_gender(perfume_gender)
            if user == perfume:
                return 1.0
            if perfume == "Unisex":
                return 0.8
            if user == "Unisex" and perfume in {"Men", "Women"}:
                return 0.7
            return 0.3


        def age_score(age, fragrance_family, aroma_profile):
            age_terms = get_age_terms(age)
            perfume_terms = tokenize_terms(fragrance_family, aroma_profile)
            if age_terms & perfume_terms:
                return 1.0
            if has_related_overlap(age_terms, perfume_terms):
                return 0.7
            return 0.4


        def rating_score(rating):
            value = parse_float(rating, 0.0)
            if value <= 0:
                return 0.0
            return min(value / 5.0, 1.0)


        def price_score(price, budget):
            value = parse_float(price, 0.0)
            if budget <= 0:
                return 0.7
            if value <= budget:
                return 1.0
            if value <= budget * 1.2:
                return 0.7
            return 0.3


        def family_score(preferred_family, aroma_keywords, row_family, aroma_profile):
            preferred = normalize_text(preferred_family)
            row = normalize_text(row_family)
            perfume_terms = tokenize_terms(row_family, aroma_profile)
            preferred_terms = tokenize_terms(preferred_family, aroma_keywords)
            if preferred and (preferred == row or preferred in row or row in preferred):
                return 1.0
            if preferred_terms and preferred_terms & perfume_terms:
                return 0.85
            if preferred_terms and has_related_overlap(preferred_terms, perfume_terms):
                return 0.7
            if preferred:
                return 0.3
            return 0.6


        def fuzzy_saw_score(scores):
            return round(sum(scores[key] * weight for key, weight in WEIGHTS.items()), 4)
        """
    ),
    md("## 10. Fungsi Rekomendasi Hybrid"),
    code(
        """
        def build_query(preference):
            age_terms = " ".join(sorted(get_age_terms(preference["age"])))
            usage_terms = USAGE_KEYWORDS.get(normalize_text(preference.get("usage", "daily")), "")
            parts = [
                preference.get("fragrance_family", ""),
                preference.get("aroma_keywords", ""),
                usage_terms,
                age_terms,
            ]
            query = " ".join(part for part in parts if clean_text(part))
            return normalize_text(query or "fresh floral woody citrus musk")


        def make_reason(row, preference):
            reasons = []
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
            if row["price_score"] >= 0.9 and preference["budget"] > 0:
                reasons.append("harga masuk budget")
            if not reasons:
                reasons.append("skor gabungan ML dan SPK paling kompetitif")
            return ", ".join(reasons).capitalize() + "."


        def recommend_perfume(
            gender="Unisex",
            age=25,
            fragrance_family="",
            aroma_keywords="",
            budget=0,
            minimal_rating=0,
            usage="daily",
            top_n=10,
        ):
            preference = {
                "gender": normalize_gender(gender),
                "age": int(age),
                "fragrance_family": clean_text(fragrance_family),
                "aroma_keywords": clean_text(aroma_keywords),
                "budget": float(budget),
                "minimal_rating": float(minimal_rating),
                "usage": clean_text(usage) or "daily",
                "top_n": int(top_n),
            }

            candidates = df.copy()
            if preference["minimal_rating"] > 0:
                filtered = candidates[candidates["rating"].ge(preference["minimal_rating"])].copy()
                if not filtered.empty:
                    candidates = filtered

            query = build_query(preference)
            query_vector = final_vectorizer.transform([query])
            candidate_matrix = final_tfidf_matrix[candidates.index]
            candidates = candidates.assign(aroma_similarity=cosine_similarity(query_vector, candidate_matrix).flatten())

            score_rows = []
            for _, row in candidates.iterrows():
                scores = {
                    "gender_score": gender_score(preference["gender"], row["gender"]),
                    "age_score": age_score(preference["age"], row["fragrance_family"], row["aroma_profile"]),
                    "aroma_similarity": float(row["aroma_similarity"]),
                    "rating_score": rating_score(row["rating"]),
                    "price_score": price_score(row["price"], preference["budget"]),
                    "family_score": family_score(
                        preference["fragrance_family"],
                        preference["aroma_keywords"],
                        row["fragrance_family"],
                        row["aroma_profile"],
                    ),
                }
                score_rows.append(scores | {"final_score": fuzzy_saw_score(scores)})

            score_df = pd.DataFrame(score_rows, index=candidates.index)
            ranked = pd.concat([candidates, score_df.drop(columns=["aroma_similarity"])], axis=1)
            ranked = ranked.sort_values(
                by=["final_score", "aroma_similarity", "rating", "rating_count"],
                ascending=[False, False, False, False],
            ).head(preference["top_n"]).copy()
            ranked["rank"] = range(1, len(ranked) + 1)
            ranked["reason"] = ranked.apply(lambda row: make_reason(row, preference), axis=1)

            columns = [
                "rank", "name", "brand", "gender", "price", "rating", "rating_count",
                "fragrance_family", "character", "top_note", "middle_note", "base_note",
                "aroma_similarity", "gender_score", "age_score", "rating_score",
                "price_score", "family_score", "final_score", "reason",
            ]
            return ranked[columns]
        """
    ),
    md("## 11. Evaluasi Fuzzy SAW dan Rule Consistency"),
    code(
        """
        # Verifikasi formula: final_score harus sama dengan jumlah skor komponen dikali bobot.
        sample_scores = {
            "gender_score": 1.0,
            "age_score": 1.0,
            "aroma_similarity": 0.8,
            "rating_score": 0.9,
            "price_score": 1.0,
            "family_score": 0.85,
        }
        manual_score = sum(sample_scores[key] * WEIGHTS[key] for key in WEIGHTS)
        function_score = fuzzy_saw_score(sample_scores)

        consistency_checks = pd.DataFrame([
            {
                "test": "Formula Fuzzy SAW",
                "expected": round(manual_score, 4),
                "actual": function_score,
                "passed": round(manual_score, 4) == function_score,
            },
            {
                "test": "Gender match lebih tinggi dari gender mismatch",
                "expected": True,
                "actual": gender_score("Men", "Men") > gender_score("Men", "Women"),
                "passed": gender_score("Men", "Men") > gender_score("Men", "Women"),
            },
            {
                "test": "Harga dalam budget lebih tinggi dari jauh di atas budget",
                "expected": True,
                "actual": price_score(400, 500) > price_score(800, 500),
                "passed": price_score(400, 500) > price_score(800, 500),
            },
            {
                "test": "Rating 5 lebih tinggi dari rating kosong",
                "expected": True,
                "actual": rating_score(5) > rating_score(0),
                "passed": rating_score(5) > rating_score(0),
            },
        ])
        consistency_checks
        """
    ),
    md("## 12. Uji Skenario Rekomendasi"),
    code(
        """
        scenario_1 = recommend_perfume(
            gender="Men",
            age=22,
            fragrance_family="Citrus",
            aroma_keywords="fresh citrus aquatic",
            budget=500,
            minimal_rating=4.0,
            usage="campus",
            top_n=5,
        )
        scenario_1
        """
    ),
    code(
        """
        scenario_2 = recommend_perfume(
            gender="Women",
            age=25,
            fragrance_family="Floral",
            aroma_keywords="floral sweet musk",
            budget=700,
            minimal_rating=4.0,
            usage="work",
            top_n=5,
        )
        scenario_2
        """
    ),
    code(
        """
        scenario_3 = recommend_perfume(
            gender="Unisex",
            age=35,
            fragrance_family="Woody",
            aroma_keywords="woody amber spicy",
            budget=1000,
            minimal_rating=4.0,
            usage="formal",
            top_n=5,
        )
        scenario_3
        """
    ),
    md("## 13. Simpan Laporan Evaluasi"),
    code(
        """
        evaluation_report = {
            "dataset": {
                "raw_rows": int(len(df_raw)),
                "clean_rows": int(len(df)),
                "train_rows": int(len(train_df)),
                "test_rows": int(len(test_df)),
                "n_tfidf_features": int(final_tfidf_matrix.shape[1]),
            },
            "retrieval_metrics": evaluation_summary.to_dict(orient="records"),
            "consistency_checks": consistency_checks.to_dict(orient="records"),
            "scenario_top_1": {
                "scenario_1": scenario_1.iloc[0][["name", "brand", "final_score"]].to_dict(),
                "scenario_2": scenario_2.iloc[0][["name", "brand", "final_score"]].to_dict(),
                "scenario_3": scenario_3.iloc[0][["name", "brand", "final_score"]].to_dict(),
            },
        }

        EVALUATION_PATH.write_text(json.dumps(evaluation_report, indent=2), encoding="utf-8")
        EVALUATION_PATH
        """
    ),
    md(
        """
        ## 14. Kesimpulan Eksperimen

        Notebook ini menghasilkan artifact yang dipakai oleh Flask:

        - `Datasets/perfume_clean.csv`
        - `models/tfidf_vectorizer.joblib`
        - `models/tfidf_matrix.joblib`
        - `models/metadata.json`
        - `reports/model_evaluation.json`

        Setelah notebook ini dijalankan, aplikasi Flask dapat memuat model hasil training,
        menghitung cosine similarity dari input user, lalu melakukan ranking akhir dengan Fuzzy SAW.
        """
    ),
]

path = Path("notebooks/01_perfume_hybrid_recommendation.ipynb")
path.parent.mkdir(exist_ok=True)
path.write_text(nbf.writes(nb), encoding="utf-8")
print(f"Wrote {path} with {len(nb.cells)} cells")
