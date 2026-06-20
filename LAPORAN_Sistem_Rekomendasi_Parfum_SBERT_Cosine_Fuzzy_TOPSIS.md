# Tugas

## Sistem Rekomendasi Parfum Berbasis Sentence-BERT, Cosine Similarity, dan Fuzzy TOPSIS (Technique for Order Preference by Similarity to Ideal Solution)

# tamabah log activiti dan export

| Komponen | Keterangan |
| --- | --- |
| Judul Project | Sistem Rekomendasi Parfum Berbasis Sentence-BERT, Cosine Similarity, dan Fuzzy TOPSIS |
| Dataset | `Datasets/final_df.csv` |
| Notebook Eksperimen | `notebooks/02_sbert_cosine_fuzzy_topsis_experiment.ipynb` |
| Framework Aplikasi | Flask |
| Metode AI | Sentence-BERT dan Cosine Similarity |
| Metode SPK | Fuzzy TOPSIS |
| Output Sistem | Ranking rekomendasi parfum sesuai preferensi pengguna |

Project ini membangun sistem rekomendasi parfum personal. Pengguna memasukkan preferensi seperti gender, umur, tujuan pemakaian, fragrance family, notes aroma, budget, rating minimal, dan jumlah hasil. Sistem kemudian mencari parfum yang paling sesuai menggunakan gabungan metode AI dan SPK.

Metode AI digunakan untuk memahami kemiripan teks aroma parfum. Metode SPK digunakan untuk menentukan ranking akhir berdasarkan banyak kriteria keputusan. Dengan begitu, hasil rekomendasi tidak hanya bergantung pada kemiripan deskripsi, tetapi juga mempertimbangkan kecocokan gender, usia, budget, rating, popularitas, dan kebutuhan pemakaian.

---

## 1. Masalah

### 1.1 Latar Belakang Masalah

Parfum adalah produk yang sangat subjektif. Dua parfum dengan rating sama belum tentu cocok untuk pengguna yang sama karena preferensi aroma dipengaruhi oleh banyak faktor, seperti:

1. Jenis aroma yang disukai.
2. Fragrance family, misalnya Floral, Woody, Oriental, Citrus, Aromatic, atau Fruity.
3. Notes parfum, seperti top note, middle note, dan base note.
4. Gender target parfum.
5. Usia pengguna.
6. Tujuan pemakaian, misalnya harian, kerja, formal, kuliah, atau date.
7. Budget pengguna.
8. Rating dan popularitas produk.

Permasalahan muncul karena dataset parfum memiliki banyak atribut tekstual. Jika sistem hanya menggunakan filter biasa, misalnya memilih family `Floral` atau harga di bawah budget tertentu, sistem belum tentu dapat memahami kemiripan aroma secara semantik. Contohnya, parfum dengan notes `jasmine`, `musk`, dan `vanilla` bisa tetap relevan untuk pengguna yang menulis preferensi `romantic sweet floral`, walaupun kata-katanya tidak selalu sama persis.

Karena itu, diperlukan sistem rekomendasi yang mampu:

1. Memahami makna teks deskripsi dan notes parfum.
2. Mengukur kemiripan antara preferensi pengguna dan profil parfum.
3. Menggabungkan similarity tersebut dengan kriteria keputusan lain.
4. Menghasilkan ranking parfum yang mudah dipahami pengguna.

### 1.2 Rumusan Masalah

Rumusan masalah pada project ini adalah:

1. Bagaimana membangun sistem rekomendasi parfum yang dapat memahami kemiripan aroma berdasarkan teks?
2. Bagaimana menggunakan Sentence-BERT untuk membentuk representasi semantik parfum?
3. Bagaimana menghitung kemiripan antara preferensi pengguna dan parfum menggunakan cosine similarity?
4. Bagaimana menggabungkan hasil similarity dengan kriteria SPK menggunakan Fuzzy TOPSIS?
5. Bagaimana menyajikan hasil rekomendasi tersebut dalam aplikasi web Flask?

### 1.3 Tujuan

Tujuan project ini adalah:

1. Membuat sistem rekomendasi parfum personal.
2. Menggunakan Sentence-BERT untuk mengubah profil parfum menjadi embedding.
3. Menggunakan cosine similarity untuk mencari parfum yang mirip dengan preferensi pengguna.
4. Menggunakan Fuzzy TOPSIS untuk ranking akhir berdasarkan beberapa kriteria.
5. Membuat aplikasi Flask agar sistem dapat digunakan melalui web.
6. Menyediakan evaluasi yang jelas melalui training loss, validation loss, pair similarity, retrieval Top-K, dan evaluasi skenario.

### 1.4 Batasan Masalah

Batasan project:

1. Dataset yang digunakan hanya `Datasets/final_df.csv`.
2. Sistem merekomendasikan parfum berdasarkan data yang tersedia di dataset.
3. Dataset belum memiliki data interaksi pengguna nyata, seperti klik, pembelian, favorit, atau rating personal.
4. Training Sentence-BERT menggunakan pseudo-label, yaitu label kemiripan yang dibuat dari atribut parfum.
5. Evaluasi retrieval menggunakan proxy relevance, bukan penilaian langsung dari pengguna.
6. Sistem belum memperhitungkan stok, lokasi pembelian, atau harga real-time.

### 1.5 Input dan Output Sistem

Input dari pengguna:

| Input | Keterangan |
| --- | --- |
| Gender | Target gender pengguna, yaitu Men, Women, atau Unisex |
| Umur | Digunakan untuk menilai kecocokan karakter aroma dengan kelompok usia |
| Pemakaian | Konteks penggunaan parfum, seperti daily, campus, work, formal, atau date |
| Fragrance family | Family aroma yang diinginkan |
| Aroma / notes | Kata kunci aroma, misalnya `fresh citrus aquatic` |
| Budget | Batas harga parfum |
| Minimal rating | Filter rating minimum |
| Jumlah hasil | Banyak rekomendasi yang ditampilkan |

Output sistem:

1. Ranking parfum terbaik.
2. Nama parfum dan brand.
3. Gambar produk.
4. Fragrance family.
5. Nilai cosine similarity.
6. Nilai akhir TOPSIS.
7. Rating dan harga.
8. Alasan rekomendasi.
9. Detail top note, middle note, dan base note.

---

## 2. Dataset

### 2.1 Sumber Dataset

Dataset utama project ini adalah:

```text
Datasets/final_df.csv
```

Dataset berisi informasi produk parfum, seperti nama produk, harga, deskripsi, rating, brand, gender, jenis produk, karakter aroma, fragrance family, ingredients, concentration, dan notes parfum.

### 2.2 Jumlah Data

| Tahap | Jumlah Data |
| --- | ---: |
| Data mentah | 4.239 |
| Data setelah preprocessing | 3.296 |

Jumlah data berkurang karena sistem hanya memakai produk parfum, menghapus kategori `Home`, dan melakukan deduplikasi berdasarkan nama, brand, dan ukuran.

### 2.3 Kolom Dataset

| Kolom Asli | Kolom Setelah Preprocessing | Keterangan |
| --- | --- | --- |
| `Name` | `name` | Nama parfum |
| `Price` | `price` | Harga parfum |
| `Description` | `description` | Deskripsi produk |
| `Rate` | `rating` | Rating produk |
| `Rating_count` | `rating_count` | Jumlah rating |
| `image` | `image` | URL gambar produk |
| `Brand` | `brand` | Brand parfum |
| `Gender` | `gender` | Target gender |
| `Product_Type` | `product_type` | Jenis produk |
| `Character_x` | `character` | Karakter parfum |
| `Fragrance_Family` | `fragrance_family` | Family aroma |
| `Size` | `size` | Ukuran produk |
| `Year` | `year` | Tahun rilis |
| `Ingredients` | `ingredients` | Bahan atau notes tambahan |
| `Concentration` | `concentration` | Konsentrasi parfum |
| `Top_note` | `top_note` | Aroma pembuka |
| `Middle_note` | `middle_note` | Aroma tengah |
| `Base_note` | `base_note` | Aroma dasar |

### 2.4 Distribusi Gender

Distribusi gender setelah preprocessing:

| Gender | Jumlah |
| --- | ---: |
| Women | 1.785 |
| Unisex | 814 |
| Men | 697 |

Berdasarkan distribusi tersebut, data parfum wanita memiliki jumlah paling banyak, disusul parfum unisex dan parfum pria.

### 2.5 Top 10 Fragrance Family

| Fragrance Family | Jumlah |
| --- | ---: |
| Floral | 1.020 |
| Oriental | 503 |
| Woody | 501 |
| Aromatic | 300 |
| Fruity | 241 |
| Citrus | 227 |
| Floral Oriental | 162 |
| Woody Oriental | 100 |
| Leather | 71 |
| Oud | 43 |

Family `Floral` merupakan family terbanyak dalam dataset. Hal ini dapat memengaruhi hasil rekomendasi karena item relevan berbasis family juga banyak berasal dari kategori tersebut.

### 2.6 Ringkasan Harga dan Rating

| Statistik | Price | Rating |
| --- | ---: | ---: |
| Count | 3.296 | 3.296 |
| Mean | 519,91 | 0,45 |
| Min | 25 | 0 |
| Median | 380 | 0 |
| Max | 8.193 | 5 |

Catatan penting: rating banyak bernilai 0 karena nilai `none` pada dataset dikonversi menjadi 0. Jadi, rating 0 tidak selalu berarti parfum buruk, tetapi bisa berarti rating tidak tersedia.

### 2.7 Preprocessing Dataset

Tahap preprocessing dilakukan agar dataset siap digunakan oleh model AI dan SPK.

Langkah preprocessing:

1. Rename kolom agar lebih konsisten.
2. Membersihkan teks kosong seperti `nan`, `none`, dan `null`.
3. Mengubah kolom numerik seperti price, rating, rating count, dan year.
4. Menormalisasi gender menjadi `Men`, `Women`, atau `Unisex`.
5. Menyaring data agar hanya mengambil produk yang mengandung kata `Perfume`.
6. Menghapus data dengan gender `Home`.
7. Menghapus duplikasi berdasarkan `name`, `brand`, dan `size`.
8. Membuat kolom `semantic_text`.

Kolom `semantic_text` adalah gabungan atribut parfum yang digunakan sebagai input Sentence-BERT.

Contoh isi `semantic_text`:

```text
Perfume name: Roberto Cavalli Paradiso.
Brand: Roberto Cavalli.
Gender: Women.
Fragrance family: Woody.
Character: Romantic.
Concentration: Eau de Parfum.
Top notes: citruses, mandarin, bergamot.
Middle notes: jasmine.
Base notes: cypress, parasol pine, pink laurel.
Ingredients: citrus, mandarin, bergamot, jasmine, pine, cypress, laurel.
Description: Woody floral fragrance...
```

Dengan bentuk ini, model dapat membaca informasi aroma secara lebih lengkap, bukan hanya nama parfum.

---

## 3. Metode AI

Metode AI pada project ini menggunakan Sentence-BERT dan cosine similarity.

### 3.1 Sentence-BERT

Sentence-BERT adalah model embedding teks berbasis Transformer. Model ini digunakan untuk mengubah teks menjadi vektor numerik. Pada project ini, setiap parfum diubah menjadi embedding berdasarkan kolom `semantic_text`.

Model dasar yang digunakan:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Alasan penggunaan model ini:

1. Ringan dan cepat untuk eksperimen.
2. Cocok untuk semantic textual similarity.
3. Menghasilkan embedding kalimat yang dapat dibandingkan menggunakan cosine similarity.
4. Dapat digunakan untuk data teks pendek sampai sedang, seperti profil parfum.

Output embedding:

| Komponen | Nilai |
| --- | --- |
| Jumlah parfum | 3.296 |
| Dimensi embedding | 384 |
| File embedding | `models/sbert_fuzzy_topsis/perfume_embeddings.npy` |

### 3.2 Pseudo-Label untuk Training

Dataset tidak memiliki label kemiripan parfum secara manual. Karena itu, dibuat pseudo-label similarity berdasarkan atribut parfum.

Komponen pseudo-label:

| Komponen | Bobot |
| --- | ---: |
| Fragrance family similarity | 0,45 |
| Notes overlap | 0,30 |
| Gender compatibility | 0,10 |
| Character similarity | 0,10 |
| Concentration similarity | 0,05 |

Rumus pseudo-label:

```text
score = 0.45 * family
      + 0.30 * notes
      + 0.10 * gender
      + 0.10 * character
      + 0.05 * concentration
```

Nilai score berada pada rentang 0 sampai 1.

Jenis pasangan data:

1. Positive pair: parfum dengan fragrance family sama atau notes yang sangat mirip.
2. Negative pair: parfum dengan fragrance family berbeda dan notes overlap rendah.

Jumlah pasangan data:

| Jenis Data | Jumlah |
| --- | ---: |
| Train pairs | 12.000 |
| Validation pairs | 2.500 |

### 3.3 Fine-Tuning Sentence-BERT

Model Sentence-BERT di-fine-tuning agar lebih sesuai dengan domain parfum.

Konfigurasi training:

| Parameter | Nilai |
| --- | --- |
| Base model | `sentence-transformers/all-MiniLM-L6-v2` |
| Loss | `CosineSimilarityLoss` |
| Batch size | 16 |
| Epoch artifact | 10 |
| Train pairs | 12.000 |
| Validation pairs | 2.500 |
| Final training loss | 0,003134 |
| Final validation loss | 0,003310 |

Loss yang digunakan adalah `CosineSimilarityLoss`. Loss ini melatih model agar cosine similarity antara dua embedding mendekati skor pseudo-label.

### 3.4 Cosine Similarity

Cosine similarity digunakan untuk menghitung kedekatan antara embedding query pengguna dan embedding parfum.

Rumus:

```text
cosine(A, B) = (A . B) / (||A|| * ||B||)
```

Keterangan:

| Simbol | Keterangan |
| --- | --- |
| `A` | Embedding query pengguna |
| `B` | Embedding parfum |
| `A . B` | Dot product |
| `||A||` dan `||B||` | Panjang vektor |

Semakin besar nilai cosine similarity, semakin mirip preferensi pengguna dengan profil parfum.

### 3.5 Pembentukan Query Pengguna

Input pengguna diubah menjadi query teks. Contoh:

| Input | Nilai |
| --- | --- |
| Gender | Women |
| Umur | 24 |
| Fragrance family | Floral |
| Aroma keywords | jasmine vanilla musk fresh romantic |
| Usage | date |

Query yang dibentuk dapat berupa:

```text
Floral jasmine vanilla musk fresh romantic floral musk romantic sweet warm Women
```

Query ini kemudian diubah menjadi embedding menggunakan model Sentence-BERT yang sudah di-fine-tuning.

### 3.6 Evaluasi Metode AI

Evaluasi pair similarity membandingkan skor pseudo-label dengan skor prediksi model.

| Metrik | Nilai |
| --- | ---: |
| Pearson | 0,9896 |
| Spearman | 0,9647 |
| MAE | 0,2791 |
| RMSE | 0,2942 |

Interpretasi:

1. Pearson 0,9896 menunjukkan hubungan linear yang sangat kuat antara pseudo-label dan prediksi model.
2. Spearman 0,9647 menunjukkan urutan similarity model sangat selaras dengan urutan pseudo-label.
3. MAE dan RMSE masih muncul karena skor cosine tidak selalu sama persis dengan skor aturan pseudo-label.

---

## 4. Metode SPK

Metode SPK yang digunakan adalah Fuzzy TOPSIS.

### 4.1 Alasan Menggunakan Fuzzy TOPSIS

Rekomendasi parfum tidak cukup hanya berdasarkan cosine similarity. Misalnya, parfum bisa memiliki deskripsi yang mirip dengan preferensi pengguna, tetapi harganya jauh di atas budget. Atau parfum cocok secara aroma, tetapi tidak cocok dengan target gender.

Karena itu, Fuzzy TOPSIS digunakan untuk menggabungkan beberapa kriteria keputusan.

Alasan penggunaan Fuzzy TOPSIS:

1. Dapat menangani banyak kriteria.
2. Setiap kriteria dapat diberi bobot.
3. Nilai kriteria dapat dibuat fuzzy pada rentang 0 sampai 1.
4. Ranking akhir didasarkan pada kedekatan dengan solusi ideal positif.
5. Cocok untuk Sistem Pendukung Keputusan.

### 4.2 Kriteria SPK

Kriteria yang digunakan:

| Kriteria | Bobot | Jenis | Keterangan |
| --- | ---: | --- | --- |
| `semantic_similarity` | 0,30 | Benefit | Kemiripan embedding SBERT |
| `family_fit` | 0,16 | Benefit | Kecocokan fragrance family |
| `gender_fit` | 0,12 | Benefit | Kecocokan gender |
| `age_fit` | 0,08 | Benefit | Kecocokan aroma dengan usia |
| `usage_fit` | 0,10 | Benefit | Kecocokan aroma dengan tujuan pemakaian |
| `rating_fit` | 0,10 | Benefit | Rating produk dan confidence |
| `popularity_fit` | 0,05 | Benefit | Popularitas berdasarkan rating count |
| `budget_fit` | 0,09 | Benefit | Kecocokan harga dengan budget |

Total bobot:

```text
0.30 + 0.16 + 0.12 + 0.08 + 0.10 + 0.10 + 0.05 + 0.09 = 1.00
```

### 4.3 Fuzzy Membership

Setiap kriteria diubah menjadi nilai fuzzy antara 0 dan 1.

Contoh aturan `budget_fit`:

| Kondisi | Nilai Fuzzy |
| --- | ---: |
| Harga <= budget | 1,00 |
| Harga <= 115% budget | 0,75 |
| Harga <= 135% budget | 0,45 |
| Harga > 135% budget | 0,15 |
| Budget tidak diisi | 0,70 |

Contoh aturan `gender_fit`:

| Kondisi | Nilai Fuzzy |
| --- | ---: |
| Gender pengguna sama dengan gender parfum | 1,00 |
| Parfum Unisex | 0,85 |
| Pengguna Unisex dan parfum Men/Women | 0,75 |
| Tidak cocok | 0,25 |

Contoh aturan `usage_fit`:

| Usage | Kata Kunci Aroma |
| --- | --- |
| daily | fresh, citrus, aquatic, clean, light, green |
| campus | fresh, citrus, fruity, aquatic, light |
| work | woody, floral, musk, aromatic, clean, elegant |
| formal | woody, amber, musk, oriental, leather, sophisticated |
| date | sweet, floral, musk, vanilla, romantic, warm |
| night | amber, oriental, musk, vanilla, woody, leather |

Nilai usage dihitung dari jumlah overlap antara kata kunci usage dan notes parfum.

### 4.4 Langkah Perhitungan TOPSIS

Langkah Fuzzy TOPSIS:

1. Membentuk matriks keputusan dari semua kandidat parfum.
2. Melakukan normalisasi matriks keputusan.
3. Mengalikan matriks normalisasi dengan bobot kriteria.
4. Menentukan solusi ideal positif.
5. Menentukan solusi ideal negatif.
6. Menghitung jarak setiap alternatif ke solusi ideal positif dan negatif.
7. Menghitung nilai preferensi atau closeness coefficient.
8. Mengurutkan parfum berdasarkan nilai TOPSIS tertinggi.

Rumus normalisasi:

```text
r_ij = x_ij / sqrt(sum(x_ij^2))
```

Rumus matriks terbobot:

```text
v_ij = w_j * r_ij
```

Solusi ideal:

```text
A+ = nilai maksimum setiap kriteria
A- = nilai minimum setiap kriteria
```

Jarak alternatif:

```text
D+ = jarak alternatif ke A+
D- = jarak alternatif ke A-
```

Skor akhir TOPSIS:

```text
C_i = D- / (D+ + D-)
```

Semakin besar nilai `C_i`, semakin baik alternatif tersebut.

### 4.5 Output Metode SPK

Output Fuzzy TOPSIS adalah `topsis_score`. Pada aplikasi, nilai ini ditampilkan sebagai skor akhir rekomendasi.

Pada backend Flask, nilai ini dipetakan menjadi:

```text
final_score = topsis_score
```

Sehingga tampilan aplikasi tetap dapat menggunakan field `final_score`.

---

## 5. Aplikasi

### 5.1 Framework

Aplikasi dibuat menggunakan Flask.

File utama aplikasi:

| File | Fungsi |
| --- | --- |
| `app.py` | Entry point aplikasi Flask |
| `app/__init__.py` | Inisialisasi Flask app |
| `app/routes.py` | Route halaman web |
| `app/recommender.py` | Logika rekomendasi |
| `app/spk.py` | Beberapa helper SPK lama dan kompatibilitas |
| `app/utils.py` | Fungsi utilitas |
| `app/templates/index.html` | Halaman input preferensi |
| `app/templates/result.html` | Halaman hasil rekomendasi |
| `app/templates/detail.html` | Halaman detail parfum |
| `app/static/css/styles.css` | Tampilan aplikasi |

### 5.2 Halaman Aplikasi

Halaman aplikasi:

1. Halaman input preferensi.
2. Halaman hasil rekomendasi.
3. Halaman detail parfum.

Pada halaman input, pengguna mengisi preferensi. Pada halaman hasil, sistem menampilkan daftar parfum dengan ranking, gambar, nama, brand, family, skor cosine, skor TOPSIS, rating, harga, dan alasan rekomendasi. Pada halaman detail, sistem menampilkan informasi lengkap parfum.

### 5.3 Integrasi Artifact Notebook ke Flask

Notebook menghasilkan artifact:

| Artifact | Fungsi |
| --- | --- |
| `models/sbert_fuzzy_topsis/fine_tuned_model` | Model Sentence-BERT hasil fine-tuning |
| `models/sbert_fuzzy_topsis/perfume_embeddings.npy` | Embedding parfum |
| `reports/sbert_fuzzy_topsis/prepared_perfume_dataset.csv` | Dataset bersih |
| `reports/sbert_fuzzy_topsis/experiment_summary.json` | Ringkasan eksperimen |
| `reports/sbert_fuzzy_topsis/retrieval_metrics.csv` | Hasil evaluasi retrieval |
| `reports/sbert_fuzzy_topsis/scenario_metrics.csv` | Hasil evaluasi skenario |

Pada saat aplikasi dijalankan, `app/recommender.py` akan memprioritaskan artifact hasil notebook:

1. Membaca `prepared_perfume_dataset.csv`.
2. Membaca `perfume_embeddings.npy`.
3. Membaca model dari `fine_tuned_model`.
4. Menggunakan Sentence-BERT untuk embedding query pengguna.
5. Menghitung cosine similarity.
6. Menghitung Fuzzy TOPSIS.
7. Mengirim hasil ranking ke template Flask.

Jika artifact SBERT tidak ditemukan, aplikasi dapat fallback ke metode lama berbasis TF-IDF agar tetap berjalan.

### 5.4 Cara Menjalankan Aplikasi

Masuk ke folder project:

```powershell
cd "f:\perkuliahan\Semester 6\SPK\project\parfum\Perfumes_Recommender-main"
```

Install dependency:

```powershell
python -m pip install -r requirements.txt
```

Jika ingin menjalankan eksperimen notebook:

```powershell
python -m pip install -r requirements-experiment.txt
```

Jika menggunakan GPU NVIDIA:

```powershell
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install -r requirements-experiment-cuda.txt
python -m pip install -r requirements-experiment.txt
python -m pip install -r requirements.txt
```

Jalankan Flask:

```powershell
python app.py
```

Buka browser:

```text
http://127.0.0.1:5000
```

---

## 6. Hasil

### 6.1 Hasil Preprocessing

| Keterangan | Jumlah |
| --- | ---: |
| Data mentah | 4.239 |
| Data bersih | 3.296 |
| Dimensi embedding | 384 |

Dataset bersih disimpan pada:

```text
reports/sbert_fuzzy_topsis/prepared_perfume_dataset.csv
```

Embedding disimpan pada:

```text
models/sbert_fuzzy_topsis/perfume_embeddings.npy
```

### 6.2 Hasil Training

| Metrik | Nilai |
| --- | ---: |
| Final training loss | 0,003134 |
| Final validation loss | 0,003310 |
| Pearson | 0,9896 |
| Spearman | 0,9647 |
| MAE | 0,2791 |
| RMSE | 0,2942 |

Interpretasi:

1. Training loss dan validation loss kecil menunjukkan model mampu mengikuti pseudo-label dengan baik.
2. Pearson 0,9896 menunjukkan hubungan linear yang sangat kuat.
3. Spearman 0,9647 menunjukkan urutan similarity model sangat mirip dengan urutan pseudo-label.
4. MAE dan RMSE masih ada karena skor cosine model tidak selalu sama persis dengan skor aturan pseudo-label.

### 6.3 Hasil Evaluasi Retrieval Top-K

Evaluasi retrieval menggunakan proxy relevance. Item dianggap relevan jika memiliki fragrance family yang sama atau family terkait dengan overlap notes yang cukup.

| K | Precision | Recall | Hit Rate | MRR | MAP | nDCG |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0,9992 | 0,0070 | 1,0000 | 1,0000 | 0,9992 | 0,9994 |
| 10 | 0,9990 | 0,0139 | 1,0000 | 1,0000 | 0,9988 | 0,9992 |
| 20 | 0,9988 | 0,0277 | 1,0000 | 1,0000 | 0,9984 | 0,9990 |

Interpretasi:

1. Precision sangat tinggi, artinya item Top-K hampir seluruhnya relevan berdasarkan proxy relevance.
2. Hit Rate 1,0000 berarti setiap query menemukan minimal satu item relevan pada Top-K.
3. MRR 1,0000 berarti item relevan pertama umumnya muncul di ranking pertama.
4. Recall rendah karena jumlah item relevan rata-rata sangat besar, yaitu sekitar 953 item relevan per query. Top-K hanya mengambil sebagian kecil dari seluruh item relevan.

### 6.4 Hasil Evaluasi Skenario

| Skenario | Mean TOPSIS | Mean Cosine | Mean Criteria | Family Hit Rate | Gender Fit Rate | Budget Fit Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Daily fresh untuk wanita muda | 0,6230 | 0,2791 | 0,6842 | 0,40 | 0,90 | 0,90 |
| Formal woody untuk pria dewasa | 0,7405 | 0,3765 | 0,7957 | 0,70 | 0,90 | 1,00 |
| Romantic floral date night | 0,7268 | 0,3743 | 0,8231 | 1,00 | 1,00 | 1,00 |
| Unisex office clean aromatic | 0,7579 | 0,3467 | 0,8005 | 1,00 | 1,00 | 1,00 |

Interpretasi:

1. Skenario `Unisex office clean aromatic` memperoleh mean TOPSIS tertinggi, yaitu 0,7579.
2. Skenario `Romantic floral date night` memperoleh family hit rate, gender fit rate, dan budget fit rate sebesar 1,00.
3. Skenario `Daily fresh untuk wanita muda` memiliki family hit rate lebih rendah, tetapi gender dan budget masih cukup baik.
4. Perbedaan nilai TOPSIS menunjukkan bahwa ranking akhir dipengaruhi oleh kombinasi banyak kriteria, bukan hanya cosine similarity.

### 6.5 Contoh Hasil Rekomendasi

Contoh input:

| Input | Nilai |
| --- | --- |
| Gender | Women |
| Umur | 24 |
| Fragrance family | Floral |
| Aroma keywords | jasmine vanilla musk fresh romantic |
| Budget | 400 |
| Minimal rating | 4 |
| Usage | date |

Contoh output rekomendasi dari sistem:

| Rank | Parfum | Cosine | TOPSIS |
| ---: | --- | ---: | ---: |
| 1 | Roberto Cavalli Paradiso | 0,3908 | 0,8302 |
| 2 | Mancera Roses Vanille | 0,2745 | 0,8234 |
| 3 | Who Am I Just Fly EDP | 0,4197 | 0,7856 |

Hasil tersebut menunjukkan bahwa ranking akhir tidak selalu hanya mengikuti cosine similarity tertinggi. Parfum dengan cosine lebih rendah tetap bisa berada di ranking tinggi jika kriteria SPK lain lebih sesuai, misalnya budget, rating, gender, family, dan usage.

### 6.6 Kesimpulan Hasil

Berdasarkan hasil eksperimen:

1. Sentence-BERT berhasil digunakan untuk membentuk representasi semantik parfum.
2. Cosine similarity mampu mengukur kedekatan query pengguna dengan profil parfum.
3. Fuzzy TOPSIS berhasil menggabungkan similarity dengan kriteria keputusan lain.
4. Sistem dapat menghasilkan ranking rekomendasi yang lebih seimbang daripada hanya menggunakan similarity.
5. Aplikasi Flask berhasil menggunakan artifact hasil notebook untuk rekomendasi.

---

## 7. Pengembangan Lanjutan

Pengembangan yang dapat dilakukan:

### 7.1 Menambahkan Data Interaksi Pengguna

Dataset saat ini belum memiliki data interaksi pengguna. Pengembangan berikutnya dapat menambahkan:

1. Data klik produk.
2. Data favorit.
3. Data pembelian.
4. Rating personal pengguna.
5. Feedback cocok atau tidak cocok.

Data tersebut dapat digunakan untuk membuat label rekomendasi yang lebih realistis.

### 7.2 Evaluasi dengan User Study

Evaluasi saat ini menggunakan proxy relevance. Agar lebih valid, sistem dapat diuji langsung kepada pengguna.

Contoh evaluasi user study:

1. Pengguna memasukkan preferensi.
2. Sistem menampilkan Top-10 rekomendasi.
3. Pengguna memberi nilai relevansi setiap rekomendasi.
4. Nilai tersebut digunakan untuk menghitung precision, recall, nDCG, dan kepuasan pengguna.

### 7.3 Perbaikan Model AI

Pengembangan model AI:

1. Membandingkan beberapa model embedding.
2. Menambah epoch dan melakukan hyperparameter tuning.
3. Mengatur ulang pseudo-label agar lebih sesuai dengan domain parfum.
4. Menggunakan hard negative mining.
5. Menggabungkan embedding teks dengan fitur numerik.

### 7.4 Perbaikan Metode SPK

Pengembangan metode SPK:

1. Menyesuaikan bobot TOPSIS berdasarkan preferensi pengguna.
2. Menambahkan metode pembobotan seperti AHP atau entropy weighting.
3. Membandingkan Fuzzy TOPSIS dengan SAW, WASPAS, VIKOR, atau MOORA.
4. Menambahkan explainability untuk setiap kriteria.

### 7.5 Pengembangan Aplikasi

Pengembangan aplikasi:

1. Menambahkan halaman riwayat rekomendasi.
2. Menambahkan filter brand, concentration, ukuran, dan tahun.
3. Menambahkan fitur simpan parfum favorit.
4. Menambahkan dashboard evaluasi model.
5. Menambahkan login pengguna.
6. Menambahkan rekomendasi berdasarkan parfum yang sedang dilihat.

---

## 8. Alur Sistem Project

### 8.1 Alur Besar Project

Alur besar project:

```text
final_df.csv
    |
    v
Preprocessing Dataset
    |
    v
Pembuatan semantic_text
    |
    v
Pembuatan pseudo-label similarity
    |
    v
Fine-tuning Sentence-BERT
    |
    v
Embedding semua parfum
    |
    v
Simpan artifact model dan embedding
    |
    v
Flask membaca artifact
    |
    v
User mengisi form preferensi
    |
    v
Query user diubah menjadi embedding
    |
    v
Cosine similarity dihitung
    |
    v
Fuzzy TOPSIS menghitung ranking akhir
    |
    v
Hasil rekomendasi ditampilkan di web
```

### 8.2 Alur Eksperimen Notebook

Notebook:

```text
notebooks/02_sbert_cosine_fuzzy_topsis_experiment.ipynb
```

Alur notebook:

1. Cek dependency.
2. Load dataset `final_df.csv`.
3. Analisis dataset.
4. Preprocessing dataset.
5. Membuat `semantic_text`.
6. Membuat pair similarity dan pseudo-label.
7. Split train dan validation.
8. Fine-tuning Sentence-BERT.
9. Plot training loss dan validation loss.
10. Evaluasi pair similarity.
11. Membuat embedding parfum.
12. Menghitung cosine similarity.
13. Menerapkan Fuzzy TOPSIS.
14. Mengevaluasi retrieval Top-K.
15. Mengevaluasi skenario rekomendasi.
16. Menyimpan artifact.

### 8.3 Alur Aplikasi Flask

Alur aplikasi:

1. Pengguna membuka halaman utama.
2. Flask menampilkan form preferensi.
3. Pengguna mengisi form.
4. Route `/recommend` menerima input.
5. Input diubah menjadi objek preference.
6. `PerfumeRecommender` membaca model dan embedding.
7. Sistem membentuk query teks dari preference.
8. Query diubah menjadi embedding menggunakan Sentence-BERT.
9. Cosine similarity dihitung terhadap seluruh embedding parfum.
10. Sistem menghitung nilai fuzzy untuk setiap kriteria.
11. Fuzzy TOPSIS menghasilkan `final_score`.
12. Data diurutkan berdasarkan `final_score`.
13. Template `result.html` menampilkan rekomendasi.
14. Pengguna dapat membuka detail parfum.

### 8.4 Alur Perhitungan Rekomendasi

Alur perhitungan untuk satu input pengguna:

```text
Input user
    |
    v
Build query teks
    |
    v
Sentence-BERT encode query
    |
    v
Cosine similarity query dengan semua parfum
    |
    v
Hitung fuzzy criteria:
semantic_similarity, family_fit, gender_fit,
age_fit, usage_fit, rating_fit, popularity_fit, budget_fit
    |
    v
Fuzzy TOPSIS
    |
    v
Ranking Top-N
```

### 8.5 Struktur Folder Penting

```text
Perfumes_Recommender-main/
|
|-- Datasets/
|   |-- final_df.csv
|
|-- notebooks/
|   |-- 02_sbert_cosine_fuzzy_topsis_experiment.ipynb
|
|-- models/
|   |-- sbert_fuzzy_topsis/
|       |-- fine_tuned_model/
|       |-- perfume_embeddings.npy
|
|-- reports/
|   |-- sbert_fuzzy_topsis/
|       |-- prepared_perfume_dataset.csv
|       |-- experiment_summary.json
|       |-- retrieval_metrics.csv
|       |-- scenario_metrics.csv
|       |-- training_history.csv
|
|-- app/
|   |-- recommender.py
|   |-- routes.py
|   |-- templates/
|   |-- static/
|
|-- app.py
|-- requirements.txt
|-- requirements-experiment.txt
|-- requirements-experiment-cuda.txt
```

### 8.6 Ringkasan Alur Akhir

Project ini terdiri dari dua bagian utama:

1. Eksperimen notebook untuk training, evaluasi, visualisasi, dan pembuatan artifact.
2. Aplikasi Flask untuk menggunakan artifact tersebut dalam bentuk sistem rekomendasi web.

Dengan alur tersebut, proses eksperimen dan proses deployment sederhana dapat dipisahkan. Notebook digunakan untuk menghasilkan model dan embedding, sedangkan Flask digunakan untuk menyajikan rekomendasi kepada pengguna.
