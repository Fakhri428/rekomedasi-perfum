---
title: PerfumeDSS
emoji: 🧴
colorFrom: pink
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# PerfumeDSS

Aplikasi rekomendasi parfum personal berbasis Flask yang mengikuti PRD `PRD_Perfume_DSS_Hybrid_ML_SPK_Flask.md`.

> **Deploy:** aplikasi ini di-deploy ke Hugging Face Spaces (SDK Docker, port 7860).
> Lihat `DEPLOY.md` untuk langkah lengkap push ke GitHub & Hugging Face.

Alur project dibuat notebook-first:

1. Jalankan notebook untuk preprocessing, training TF-IDF, evaluasi model, dan export artifact.
2. Flask membaca artifact hasil notebook untuk proses rekomendasi web.

Notebook eksperimen tersedia di:

```text
notebooks/01_perfume_hybrid_recommendation.ipynb
```

Artifact hasil training notebook:

```text
Datasets/perfume_clean.csv
models/tfidf_vectorizer.joblib
models/tfidf_matrix.joblib
models/metadata.json
reports/model_evaluation.json
```

## Dataset

Dataset utama:

```text
Datasets/final_df.csv
```

Kolom yang dipakai:

- `Name`, `Brand`, `Gender`, `Product_Type`
- `Price`, `Rate`, `Rating_count`
- `Fragrance_Family`, `Character_x`
- `Top_note`, `Middle_note`, `Base_note`, `Ingredients`
- `Description`, `image`, `Size`, `Year`, `Concentration`

Data diproses dengan aturan:

- Hanya produk dengan `Product_Type` mengandung `Perfume`.
- `Gender=Home` dikeluarkan.
- `Rate=none` diubah menjadi 0.
- Data duplikat berdasarkan `Name`, `Brand`, dan `Size` dihapus.
- Fitur `aroma_profile` dibuat dari fragrance family, character, notes, ingredients, dan description.

## Training dan Evaluasi

Di notebook, proses training dilakukan dengan:

```python
final_vectorizer.fit_transform(df["aroma_profile"])
```

Karena sistem ini adalah recommender tanpa label supervised `cocok/tidak cocok`, evaluasi model memakai proxy `fragrance_family` dan metrik retrieval:

- Precision@K
- Recall@K
- HitRate@K
- MRR@K
- nDCG@K
- Random baseline sebagai pembanding

Hasil evaluasi tersimpan di:

```text
reports/model_evaluation.json
```

## Metode Aplikasi

Sistem memakai dua tahap:

1. Machine Learning klasik:
   TF-IDF mengubah `aroma_profile` menjadi vektor, lalu cosine similarity menghitung kemiripan input aroma pengguna dengan parfum.

2. SPK:
   Fuzzy SAW menghitung skor akhir dari 6 kriteria:
   `gender_score`, `age_score`, `aroma_similarity`, `rating_score`, `price_score`, dan `family_score`.

Bobot default:

| Kriteria | Bobot |
|---|---:|
| Gender | 0.20 |
| Umur | 0.15 |
| Aroma similarity | 0.25 |
| Rating | 0.15 |
| Harga | 0.10 |
| Fragrance family | 0.15 |

## Menjalankan Aplikasi

Install dependensi:

```bash
pip install -r requirements.txt
```

Jalankan notebook training terlebih dahulu dari Jupyter, atau rebuild notebook dengan:

```bash
python tools/build_training_notebook.py
```

Lalu execute notebook:

```bash
jupyter notebook notebooks/01_perfume_hybrid_recommendation.ipynb
```

Jalankan Flask:

```bash
python app.py
```

Buka:

```text
http://127.0.0.1:5000
```

## Struktur

```text
app.py
app/
  __init__.py
  recommender.py
  routes.py
  spk.py
  utils.py
  static/css/styles.css
  templates/base.html
  templates/index.html
  templates/result.html
  templates/detail.html
notebooks/01_perfume_hybrid_recommendation.ipynb
tools/build_training_notebook.py
Datasets/final_df.csv
Datasets/perfume_clean.csv
models/tfidf_vectorizer.joblib
models/tfidf_matrix.joblib
reports/model_evaluation.json
```

## Smoke Test

Contoh uji cepat:

```bash
python -c "from app.recommender import get_recommender, preference_from_mapping; rec=get_recommender(); pref=preference_from_mapping({'gender':'Men','age':'22','fragrance_family':'Citrus','aroma_keywords':'fresh citrus aquatic','budget':'500','minimal_rating':'4','usage':'campus','top_n':'5'}); print(rec.recommend(pref)[0])"
```
