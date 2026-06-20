# PRD — Sistem Rekomendasi Parfum Personal Berbasis Hybrid Machine Learning dan SPK dengan Flask

## 1. Informasi Project

**Nama Project:** Perfume Decision Support System  
**Nama Singkat:** PerfumeDSS  
**Jenis Project:** Hybrid Recommendation System  
**Pendekatan Utama:** Machine Learning Klasik + Sistem Pendukung Keputusan  
**Model Machine Learning:** TF-IDF + Cosine Similarity  
**Metode SPK:** Fuzzy SAW  
**Output Awal:** Notebook eksperimen `.ipynb`  
**Output Akhir:** Aplikasi web rekomendasi parfum berbasis Flask  
**Dataset Rekomendasi:** GoldenScent Perfumes Dataset / dataset parfum publik sejenis  

---

## 2. Ringkasan Project

Project ini bertujuan membangun sistem rekomendasi parfum personal yang menggabungkan **Machine Learning klasik** dan **Sistem Pendukung Keputusan (SPK)**.

Machine Learning digunakan untuk menghitung kemiripan aroma antara input pengguna dengan data parfum menggunakan **TF-IDF dan Cosine Similarity**. Sementara itu, SPK digunakan untuk menentukan ranking akhir parfum terbaik menggunakan metode **Fuzzy SAW** berdasarkan beberapa kriteria seperti jenis kelamin, umur, jenis aroma, harga, rating, dan karakter aroma.

Dengan pendekatan ini, sistem tidak hanya mencari parfum yang mirip secara aroma, tetapi juga mempertimbangkan faktor keputusan lain agar rekomendasi lebih personal dan relevan.

---

## 3. Latar Belakang

Parfum merupakan produk yang sangat personal karena pemilihannya dipengaruhi oleh banyak faktor, seperti jenis kelamin, umur, jenis aroma, harga, rating, karakter aroma, dan kebutuhan penggunaan. Banyak orang kesulitan memilih parfum yang sesuai karena jumlah pilihan sangat banyak dan informasi parfum sering tersebar dalam bentuk notes, fragrance family, review, serta rating.

Pemilihan parfum tidak cukup hanya berdasarkan rating tertinggi. Parfum dengan rating tinggi belum tentu cocok untuk semua orang, karena preferensi aroma setiap pengguna berbeda. Oleh karena itu, dibutuhkan sistem yang dapat menggabungkan analisis kemiripan aroma dan pengambilan keputusan berbasis banyak kriteria.

Project ini diawali dengan eksperimen data pada file `.ipynb` untuk melakukan data understanding, preprocessing, exploratory data analysis, ekstraksi fitur teks aroma, perhitungan TF-IDF, Cosine Similarity, dan Fuzzy SAW. Setelah metode rekomendasi berhasil diuji, sistem dikembangkan menjadi aplikasi web menggunakan Flask agar dapat digunakan secara interaktif oleh pengguna.

---

## 4. Tujuan Project

Tujuan utama project ini adalah membangun sistem rekomendasi parfum personal berbasis web dengan menggabungkan Machine Learning klasik dan Sistem Pendukung Keputusan.

Tujuan khusus:

1. Mengumpulkan dan memahami dataset parfum publik.
2. Melakukan preprocessing data parfum agar siap digunakan.
3. Mengolah data teks parfum seperti fragrance family, character, top notes, middle notes, dan base notes.
4. Mengimplementasikan TF-IDF untuk mengubah teks aroma parfum menjadi representasi numerik.
5. Mengimplementasikan Cosine Similarity untuk menghitung kemiripan antara preferensi aroma pengguna dan data parfum.
6. Menentukan kriteria SPK yang relevan untuk rekomendasi parfum.
7. Membuat rule kesesuaian umur terhadap jenis aroma parfum.
8. Mengimplementasikan metode Fuzzy SAW untuk menghasilkan ranking rekomendasi.
9. Membuat eksperimen awal dalam notebook `.ipynb`.
10. Mengembangkan hasil eksperimen menjadi aplikasi web rekomendasi parfum menggunakan Flask.
11. Menampilkan hasil rekomendasi secara informatif, lengkap dengan skor dan alasan rekomendasi.

---

## 5. Rumusan Masalah

1. Bagaimana memanfaatkan dataset parfum publik untuk membangun sistem rekomendasi parfum personal?
2. Bagaimana mengolah data teks parfum seperti fragrance family dan notes agar dapat dihitung kemiripannya?
3. Bagaimana menggunakan TF-IDF dan Cosine Similarity untuk menghitung kecocokan aroma parfum?
4. Bagaimana menentukan kriteria SPK yang sesuai untuk rekomendasi parfum?
5. Bagaimana metode Fuzzy SAW dapat digunakan untuk memberi ranking parfum terbaik?
6. Bagaimana hasil eksperimen notebook dapat dikembangkan menjadi aplikasi web berbasis Flask?

---

## 6. Jenis Project

Project ini **bukan klasifikasi utama**.

Project ini lebih tepat disebut:

> **Hybrid Recommendation System berbasis Machine Learning klasik dan Sistem Pendukung Keputusan.**

Penjelasan:

| Komponen | Jenis | Fungsi |
|---|---|---|
| TF-IDF | Machine Learning klasik / NLP klasik | Mengubah teks aroma menjadi vektor angka |
| Cosine Similarity | Similarity-based recommendation | Menghitung kemiripan aroma |
| Rule-Based Scoring | Rule-based system | Menghitung kesesuaian gender, umur, harga, dan rating |
| Fuzzy SAW | Sistem Pendukung Keputusan | Menghasilkan ranking akhir rekomendasi parfum |
| Flask | Web framework | Menyediakan aplikasi rekomendasi berbasis web |

Klasifikasi hanya digunakan secara kecil dalam bentuk rule, misalnya mengelompokkan umur ke kategori tertentu. Namun output utama sistem adalah **ranking rekomendasi parfum**, bukan label klasifikasi.

---

## 7. Ruang Lingkup Project

### 7.1 Termasuk dalam Project

Project ini mencakup:

- Penggunaan dataset parfum publik.
- Data cleaning dan preprocessing.
- Exploratory Data Analysis sederhana.
- Ekstraksi fitur teks aroma.
- Implementasi TF-IDF.
- Implementasi Cosine Similarity.
- Perancangan kriteria SPK.
- Pembuatan rule umur terhadap jenis aroma parfum.
- Implementasi Fuzzy SAW.
- Pembuatan aplikasi Flask.
- Form input preferensi pengguna.
- Output ranking rekomendasi parfum.
- Halaman detail parfum.
- Penjelasan alasan rekomendasi.

### 7.2 Tidak Termasuk dalam Project

Project ini tidak mencakup:

- Sistem pembelian parfum secara langsung.
- Integrasi marketplace.
- Login dan registrasi pengguna pada versi awal.
- Rekomendasi berbasis riwayat pembelian pengguna.
- Deep Learning pada versi awal.
- Transformer seperti BERT pada versi awal.
- Scraping data otomatis pada versi pertama.

---

## 8. Target Pengguna

### 8.1 Pengguna Umum

Pengguna yang ingin mencari parfum berdasarkan preferensi pribadi, seperti gender, umur, aroma, harga, dan rating.

### 8.2 Mahasiswa / Peneliti

Pengguna yang ingin mempelajari penerapan hybrid recommendation system yang menggabungkan Machine Learning klasik dan SPK.

### 8.3 Penjual Parfum

Penjual parfum yang ingin memberikan rekomendasi produk kepada pelanggan berdasarkan kriteria tertentu.

---

## 9. Dataset

### 9.1 Dataset Utama

Dataset yang direkomendasikan adalah dataset parfum publik seperti:

**GoldenScent Perfumes Dataset**

Dataset ini cocok karena umumnya memuat atribut seperti:

- Nama parfum
- Brand
- Harga
- Gender
- Fragrance family
- Character
- Rating
- Top notes
- Middle notes
- Base notes
- Review

### 9.2 Alternatif Dataset

Jika GoldenScent tidak lengkap atau sulit digunakan, alternatif dataset:

1. Fragrantica Fragrance Dataset
2. Parfumo Dataset
3. Kaggle Perfume Dataset
4. Dataset parfum dari GitHub atau Hugging Face

### 9.3 Kolom yang Digunakan

| Kolom Dataset | Fungsi dalam Sistem |
|---|---|
| Nama parfum | Alternatif rekomendasi |
| Brand | Informasi parfum |
| Gender | Kriteria kesesuaian gender |
| Price | Kriteria harga |
| Average Rating | Kriteria rating |
| Fragrance Family | Fitur aroma dan kriteria SPK |
| Character | Fitur aroma tambahan |
| Top Notes | Fitur teks untuk TF-IDF |
| Middle Notes | Fitur teks untuk TF-IDF |
| Base Notes | Fitur teks untuk TF-IDF |
| Reviews | Opsional untuk pengembangan sentiment analysis |

### 9.4 Catatan Umur

Variabel **umur** biasanya tidak tersedia langsung dalam dataset parfum. Oleh karena itu, umur dibuat sebagai rule tambahan dalam sistem.

Contoh rule umur:

| Kelompok Umur | Jenis Aroma yang Cocok |
|---|---|
| 15–20 tahun | Fresh, Citrus, Fruity, Sweet |
| 21–30 tahun | Fresh, Aquatic, Floral, Woody, Sweet |
| 31–45 tahun | Woody, Spicy, Amber, Musk |
| 46+ tahun | Oriental, Leather, Amber, Powdery, Musk |

---

## 10. Model Machine Learning yang Digunakan

### 10.1 TF-IDF

TF-IDF digunakan untuk mengubah teks parfum menjadi representasi angka atau vektor.

Data teks yang digunakan:

- Fragrance family
- Character
- Top notes
- Middle notes
- Base notes

Contoh gabungan fitur teks parfum:

```text
fresh citrus bergamot aquatic musk woody aromatic
```

TF-IDF akan menghitung bobot kata berdasarkan seberapa penting kata tersebut di dalam data parfum.

### 10.2 Cosine Similarity

Cosine Similarity digunakan untuk menghitung tingkat kemiripan antara input aroma pengguna dengan data aroma parfum.

Contoh:

Input user:

```text
fresh citrus aquatic
```

Data parfum:

```text
fresh bergamot aquatic musk
```

Sistem akan menghitung skor kemiripan antara input user dan parfum tersebut.

Output:

```text
aroma_similarity_score = 0.85
```

Skor ini kemudian digunakan sebagai salah satu kriteria dalam Fuzzy SAW.

### 10.3 Alasan Menggunakan TF-IDF + Cosine Similarity

Model ini dipilih karena:

1. Dataset parfum banyak berisi teks pendek seperti notes dan fragrance family.
2. Implementasinya ringan dan cocok untuk notebook serta Flask.
3. Mudah dijelaskan dalam laporan.
4. Cocok untuk sistem rekomendasi berbasis kemiripan.
5. Tidak membutuhkan dataset label seperti cocok/tidak cocok.
6. Lebih sederhana dibanding Transformer atau Deep Learning.

### 10.4 Kenapa Tidak Menggunakan Deep Learning?

Deep Learning belum digunakan pada versi awal karena:

1. Dataset belum tentu memiliki label yang cukup.
2. Komputasi lebih berat.
3. Interpretasi lebih sulit.
4. Tujuan awal adalah membangun sistem rekomendasi yang ringan dan mudah dijelaskan.
5. TF-IDF + Cosine Similarity sudah cukup untuk menghitung kemiripan notes parfum.

Deep Learning seperti BERT dapat digunakan pada pengembangan lanjutan, terutama untuk sentiment analysis review parfum.

---

## 11. Sistem Pendukung Keputusan yang Digunakan

### 11.1 Metode SPK

Metode SPK yang digunakan adalah:

> **Fuzzy SAW**

Fuzzy SAW dipilih karena preferensi parfum bersifat subjektif. Nilai seperti cocok, cukup cocok, murah, sedang, mahal, fresh, woody, atau sweet dapat diubah menjadi nilai fuzzy agar lebih fleksibel.

### 11.2 Kriteria SPK

| Kode | Kriteria | Tipe | Sumber |
|---|---|---|---|
| C1 | Kesesuaian Gender | Benefit | Dataset + input user |
| C2 | Kesesuaian Umur | Benefit | Rule umur |
| C3 | Kemiripan Aroma | Benefit | TF-IDF + Cosine Similarity |
| C4 | Rating Parfum | Benefit | Dataset |
| C5 | Harga | Cost | Dataset + input budget |
| C6 | Kesesuaian Fragrance Family | Benefit | Dataset + input user |

### 11.3 Bobot Kriteria Awal

| Kode | Kriteria | Bobot |
|---|---|---|
| C1 | Kesesuaian Gender | 0.20 |
| C2 | Kesesuaian Umur | 0.15 |
| C3 | Kemiripan Aroma | 0.25 |
| C4 | Rating Parfum | 0.15 |
| C5 | Harga | 0.10 |
| C6 | Kesesuaian Fragrance Family | 0.15 |

Total bobot = 1.00.

Bobot dapat disesuaikan berdasarkan kebutuhan penelitian atau preferensi pengguna.

---

## 12. Logika Hybrid Model

Sistem ini menggunakan dua tahap utama:

### 12.1 Tahap Machine Learning

Pada tahap ini, sistem menghitung kemiripan aroma.

```text
Input aroma user
        ↓
Gabungan teks fragrance family + character + notes parfum
        ↓
TF-IDF Vectorization
        ↓
Cosine Similarity
        ↓
aroma_similarity_score
```

### 12.2 Tahap SPK

Pada tahap ini, sistem menghitung ranking akhir.

```text
gender_score
age_score
aroma_similarity_score
rating_score
price_score
fragrance_family_score
        ↓
Fuzzy SAW
        ↓
final_score
        ↓
Ranking parfum
```

### 12.3 Alur Gabungan

```text
Dataset Parfum
        ↓
Preprocessing
        ↓
Ekstraksi Fitur Teks Aroma
        ↓
TF-IDF + Cosine Similarity
        ↓
Skor Kemiripan Aroma
        ↓
Rule-Based Scoring
        ↓
Fuzzy SAW
        ↓
Ranking Rekomendasi Parfum
        ↓
Aplikasi Flask
```

---

## 13. Rumus Perhitungan

### 13.1 Skor Gender

| Kondisi | Skor |
|---|---|
| Gender parfum sama dengan input user | 1.0 |
| Parfum unisex | 0.8 |
| Gender berbeda | 0.3 |

### 13.2 Skor Umur

| Kondisi | Skor |
|---|---|
| Aroma parfum cocok dengan kelompok umur | 1.0 |
| Aroma cukup dekat | 0.7 |
| Aroma kurang cocok | 0.4 |

### 13.3 Skor Aroma dari ML

Skor aroma diperoleh dari Cosine Similarity:

```text
aroma_similarity_score = cosine_similarity(input_user_vector, perfume_vector)
```

Nilai berada pada rentang 0 sampai 1.

### 13.4 Skor Rating

Rating dinormalisasi dengan rumus:

```text
rating_score = rating / 5
```

### 13.5 Skor Harga

Karena harga adalah kriteria cost, semakin sesuai dengan budget maka semakin baik.

| Kondisi | Skor |
|---|---|
| Harga di bawah atau sama dengan budget | 1.0 |
| Harga 1–20% di atas budget | 0.7 |
| Harga lebih dari 20% di atas budget | 0.3 |

### 13.6 Skor Fragrance Family

| Kondisi | Skor |
|---|---|
| Fragrance family sama dengan input user | 1.0 |
| Masih satu kelompok aroma | 0.7 |
| Tidak sesuai | 0.3 |

### 13.7 Rumus Fuzzy SAW

Nilai akhir:

```text
Final Score =
(C1 × W1) +
(C2 × W2) +
(C3 × W3) +
(C4 × W4) +
(C5 × W5) +
(C6 × W6)
```

Contoh:

```text
Final Score =
(gender_score × 0.20) +
(age_score × 0.15) +
(aroma_similarity_score × 0.25) +
(rating_score × 0.15) +
(price_score × 0.10) +
(fragrance_family_score × 0.15)
```

---

## 14. Alur Notebook `.ipynb`

1. Import library.
2. Load dataset parfum.
3. Melihat struktur dataset.
4. Memilih kolom yang relevan.
5. Membersihkan data kosong.
6. Menstandarkan nama kolom.
7. Membersihkan data harga dan rating.
8. Mengolah kolom gender.
9. Mengolah fragrance family, character, dan notes.
10. Membuat fitur teks gabungan aroma.
11. Membuat rule umur.
12. Mengimplementasikan TF-IDF.
13. Menghitung Cosine Similarity.
14. Menentukan kriteria dan bobot Fuzzy SAW.
15. Menghitung skor gender, umur, aroma, rating, harga, dan fragrance family.
16. Menghitung final score.
17. Menampilkan top-N rekomendasi.
18. Menguji beberapa skenario pengguna.
19. Menyimpan dataset bersih.
20. Menyimpan fungsi atau modul rekomendasi untuk digunakan di Flask.

---

## 15. Alur Aplikasi Flask

1. Pengguna membuka halaman utama.
2. Pengguna mengisi form preferensi:
   - Jenis kelamin
   - Umur
   - Jenis aroma
   - Budget
   - Minimal rating
   - Kebutuhan pemakaian
3. Sistem membaca data parfum yang sudah diproses.
4. Sistem membentuk input aroma pengguna.
5. Sistem menghitung TF-IDF dan Cosine Similarity.
6. Sistem menghitung skor rule-based.
7. Sistem menghitung final score menggunakan Fuzzy SAW.
8. Sistem mengurutkan parfum berdasarkan skor tertinggi.
9. Sistem menampilkan rekomendasi parfum.
10. Pengguna dapat melihat detail parfum.

---

## 16. User Flow

```text
User membuka aplikasi
        ↓
User mengisi form preferensi parfum
        ↓
Sistem memvalidasi input
        ↓
Sistem mencocokkan input dengan dataset
        ↓
TF-IDF mengubah teks aroma menjadi vektor
        ↓
Cosine Similarity menghitung kemiripan aroma
        ↓
Rule menghitung skor gender, umur, harga, rating
        ↓
Fuzzy SAW menghitung skor akhir
        ↓
User melihat ranking rekomendasi parfum terbaik
```

---

## 17. Fitur Utama

### 17.1 Fitur Notebook

| Fitur | Deskripsi |
|---|---|
| Load Dataset | Membaca dataset parfum |
| Data Cleaning | Membersihkan data kosong dan format tidak sesuai |
| EDA | Melihat distribusi brand, gender, rating, harga, dan aroma |
| Feature Engineering | Menggabungkan fragrance family, character, dan notes |
| TF-IDF | Mengubah teks aroma menjadi vektor |
| Cosine Similarity | Menghitung kemiripan aroma |
| Rule Umur | Membuat aturan umur terhadap jenis aroma |
| Fuzzy SAW | Menghitung skor akhir rekomendasi |
| Ranking | Mengurutkan parfum terbaik |
| Export Data | Menyimpan data hasil preprocessing |

### 17.2 Fitur Aplikasi Flask

| Fitur | Deskripsi |
|---|---|
| Homepage | Halaman pembuka aplikasi |
| Form Rekomendasi | Input gender, umur, aroma, budget, rating |
| Proses ML | Menghitung kemiripan aroma dengan TF-IDF + Cosine Similarity |
| Proses SPK | Menghitung ranking akhir dengan Fuzzy SAW |
| Hasil Rekomendasi | Menampilkan ranking parfum |
| Detail Parfum | Menampilkan nama, brand, rating, harga, notes |
| Alasan Rekomendasi | Menjelaskan kenapa parfum direkomendasikan |
| Filter Data | Filter berdasarkan brand, gender, harga, dan aroma |

---

## 18. Input Pengguna

| Input | Tipe | Contoh |
|---|---|---|
| Jenis Kelamin | Pilihan | Pria, Wanita, Unisex |
| Umur | Angka | 21 |
| Jenis Aroma | Pilihan | Fresh, Woody, Sweet |
| Budget | Angka | 500000 |
| Minimal Rating | Angka | 4.0 |
| Kebutuhan Pemakaian | Pilihan | Harian, Kuliah, Kerja, Formal, Date |

---

## 19. Output Sistem

| Output | Deskripsi |
|---|---|
| Ranking | Urutan rekomendasi parfum |
| Nama Parfum | Nama parfum yang direkomendasikan |
| Brand | Brand parfum |
| Harga | Harga parfum |
| Rating | Rating parfum |
| Fragrance Family | Jenis aroma parfum |
| Notes | Top, middle, dan base notes |
| Aroma Similarity | Skor kemiripan aroma dari ML |
| Skor SPK | Nilai akhir hasil Fuzzy SAW |
| Alasan | Penjelasan singkat kenapa parfum cocok |

Contoh output:

| Ranking | Parfum | Brand | Aroma Similarity | Final Score | Alasan |
|---|---|---|---|---|---|
| 1 | Perfume A | Brand X | 0.89 | 0.91 | Cocok dengan gender, umur, aroma fresh, rating tinggi |
| 2 | Perfume B | Brand Y | 0.84 | 0.87 | Aroma woody sesuai preferensi dan harga masih sesuai budget |
| 3 | Perfume C | Brand Z | 0.80 | 0.82 | Rating tinggi dan cocok untuk penggunaan harian |

---

## 20. Desain Arsitektur Sistem

```text
Dataset Parfum
     ↓
Notebook Preprocessing dan Eksperimen
     ↓
TF-IDF Vectorizer
     ↓
Cosine Similarity
     ↓
Fuzzy SAW Ranking
     ↓
Data Bersih / Modul Rekomendasi
     ↓
Backend Flask
     ↓
Form Input Pengguna
     ↓
Ranking Rekomendasi
     ↓
Tampilan Web
```

---

## 21. Struktur Folder Project

```text
perfume-dss/
│
├── data/
│   ├── raw/
│   │   └── perfume_dataset.csv
│   ├── processed/
│   │   └── perfume_clean.csv
│
├── notebooks/
│   └── 01_perfume_hybrid_recommendation.ipynb
│
├── app/
│   ├── static/
│   │   ├── css/
│   │   └── images/
│   ├── templates/
│   │   ├── index.html
│   │   ├── result.html
│   │   └── detail.html
│   ├── routes.py
│   ├── recommender.py
│   ├── spk.py
│   └── utils.py
│
├── app.py
├── requirements.txt
├── README.md
└── PRD_Perfume_DSS_Hybrid_ML_SPK_Flask.md
```

---

## 22. Rancangan Modul

### 22.1 Modul Data Preprocessing

Fungsi:

- Membersihkan kolom harga.
- Membersihkan kolom rating.
- Mengisi data kosong.
- Menstandarkan gender.
- Menstandarkan fragrance family.
- Menghapus duplikasi.
- Membuat kolom gabungan teks aroma.
- Menyimpan dataset bersih.

### 22.2 Modul Machine Learning

Fungsi:

- Membuat corpus dari kolom fragrance family, character, dan notes.
- Mengubah corpus menjadi vektor menggunakan TF-IDF.
- Mengubah input aroma pengguna menjadi vektor.
- Menghitung Cosine Similarity.
- Menghasilkan skor kemiripan aroma.

### 22.3 Modul Rule Umur

Fungsi:

- Mengubah umur menjadi kelompok umur.
- Menentukan aroma yang cocok untuk setiap kelompok umur.
- Menghasilkan skor kesesuaian umur.

### 22.4 Modul SPK

Fungsi:

- Menghitung skor gender.
- Menghitung skor umur.
- Menggunakan skor aroma dari Cosine Similarity.
- Menghitung skor rating.
- Menghitung skor harga.
- Menghitung skor fragrance family.
- Melakukan perhitungan Fuzzy SAW.
- Mengurutkan hasil rekomendasi.

### 22.5 Modul Flask

Fungsi:

- Menampilkan halaman utama.
- Menerima input pengguna.
- Mengirim input ke modul rekomendasi.
- Menampilkan hasil rekomendasi.
- Menampilkan detail parfum.

---

## 23. Rancangan Halaman Aplikasi

### 23.1 Halaman Utama

Isi:

- Judul aplikasi.
- Deskripsi singkat.
- Form input rekomendasi.
- Tombol cari rekomendasi.

### 23.2 Halaman Hasil Rekomendasi

Isi:

- Ringkasan input pengguna.
- Daftar ranking parfum.
- Aroma similarity score.
- Final score.
- Alasan rekomendasi.
- Tombol lihat detail.

### 23.3 Halaman Detail Parfum

Isi:

- Nama parfum.
- Brand.
- Gender.
- Harga.
- Rating.
- Fragrance family.
- Top notes.
- Middle notes.
- Base notes.
- Character.
- Review singkat jika tersedia.

---

## 24. Tech Stack

### 24.1 Eksperimen Data

| Kebutuhan | Tools |
|---|---|
| Bahasa | Python |
| Notebook | Jupyter Notebook |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| TF-IDF | TfidfVectorizer |
| Similarity | Cosine Similarity |
| Visualisasi | Matplotlib, Seaborn |
| Export Data | CSV / Pickle / Joblib |

### 24.2 Aplikasi Web

| Kebutuhan | Tools |
|---|---|
| Backend | Flask |
| Frontend | HTML, CSS, Bootstrap |
| Template Engine | Jinja2 |
| Data | CSV / SQLite |
| Model File | Pickle / Joblib |
| Deployment Opsional | Render, Railway, PythonAnywhere |

---

## 25. Kebutuhan Fungsional

| Kode | Kebutuhan |
|---|---|
| FR-01 | Sistem dapat membaca dataset parfum |
| FR-02 | Sistem dapat membersihkan dataset |
| FR-03 | Sistem dapat membuat fitur teks aroma |
| FR-04 | Sistem dapat menghitung TF-IDF |
| FR-05 | Sistem dapat menghitung Cosine Similarity |
| FR-06 | Sistem dapat menerima input gender |
| FR-07 | Sistem dapat menerima input umur |
| FR-08 | Sistem dapat menerima input jenis aroma |
| FR-09 | Sistem dapat menerima input budget |
| FR-10 | Sistem dapat menerima input minimal rating |
| FR-11 | Sistem dapat menghitung skor Fuzzy SAW |
| FR-12 | Sistem dapat membuat ranking rekomendasi |
| FR-13 | Sistem dapat menampilkan detail parfum |
| FR-14 | Sistem dapat menjelaskan alasan rekomendasi |
| FR-15 | Sistem dapat memfilter hasil berdasarkan kriteria |

---

## 26. Kebutuhan Non-Fungsional

| Kode | Kebutuhan |
|---|---|
| NFR-01 | Aplikasi mudah digunakan |
| NFR-02 | Waktu proses rekomendasi cepat |
| NFR-03 | Tampilan sederhana dan responsif |
| NFR-04 | Data mudah diperbarui |
| NFR-05 | Sistem dapat dijalankan secara lokal |
| NFR-06 | Kode mudah dikembangkan |
| NFR-07 | Perhitungan rekomendasi transparan dan dapat dijelaskan |
| NFR-08 | Model ringan dan tidak membutuhkan GPU |

---

## 27. Tahapan Pengembangan

### Tahap 1 — Persiapan Dataset

- Download dataset parfum.
- Simpan dataset di folder `data/raw`.
- Pelajari struktur kolom.
- Pilih kolom yang digunakan.

### Tahap 2 — Notebook Eksperimen

- Load dataset.
- Cleaning data.
- EDA sederhana.
- Buat kolom gabungan aroma.
- Implementasi TF-IDF.
- Implementasi Cosine Similarity.
- Buat rule umur.
- Buat bobot kriteria.
- Implementasi Fuzzy SAW.
- Uji beberapa input pengguna.
- Simpan dataset bersih dan model TF-IDF.

### Tahap 3 — Backend Flask

- Buat struktur project Flask.
- Buat route halaman utama.
- Buat route hasil rekomendasi.
- Buat modul `recommender.py`.
- Buat modul `spk.py`.
- Hubungkan form dengan proses rekomendasi.

### Tahap 4 — Frontend

- Buat tampilan homepage.
- Buat form input.
- Buat tampilan hasil ranking.
- Buat tampilan detail parfum.
- Tambahkan Bootstrap agar responsif.

### Tahap 5 — Testing

- Uji input berbeda.
- Cek apakah hasil ranking masuk akal.
- Cek data kosong.
- Cek harga dan rating.
- Cek hasil Cosine Similarity.
- Cek perhitungan Fuzzy SAW.
- Cek tampilan halaman.

### Tahap 6 — Finalisasi

- Rapikan kode.
- Buat README.
- Buat dokumentasi cara menjalankan.
- Siapkan laporan atau presentasi.

---

## 28. Contoh Skenario Pengguna

### Skenario 1

**Input User:**

- Gender: Pria
- Umur: 22
- Aroma: Fresh Citrus Aquatic
- Budget: 500000
- Minimal Rating: 4.0
- Kebutuhan: Kuliah / harian

**Proses Sistem:**

- TF-IDF mengubah input aroma user menjadi vektor.
- Cosine Similarity mencari parfum dengan notes paling mirip.
- Rule menghitung kesesuaian gender, umur, harga, dan rating.
- Fuzzy SAW menghitung skor akhir.

**Output Sistem:**

Sistem menampilkan parfum dengan aroma fresh, citrus, aquatic, atau woody ringan yang cocok untuk pria usia dewasa muda, memiliki rating tinggi, dan harga sesuai budget.

### Skenario 2

**Input User:**

- Gender: Wanita
- Umur: 25
- Aroma: Floral Sweet Musk
- Budget: 700000
- Minimal Rating: 4.2
- Kebutuhan: Kerja / formal

**Output Sistem:**

Sistem menampilkan parfum floral, musk, amber, atau sweet yang cocok untuk wanita dewasa muda dengan karakter elegan dan rating tinggi.

### Skenario 3

**Input User:**

- Gender: Unisex
- Umur: 35
- Aroma: Woody Amber Spicy
- Budget: 1000000
- Minimal Rating: 4.3
- Kebutuhan: Formal

**Output Sistem:**

Sistem menampilkan parfum woody, spicy, amber, atau musk dengan karakter lebih dewasa, elegan, dan cocok untuk acara formal.

---

## 29. Risiko dan Solusi

| Risiko | Solusi |
|---|---|
| Dataset tidak memiliki kolom umur | Buat rule umur manual |
| Dataset tidak lengkap | Gunakan kolom alternatif seperti notes dan fragrance family |
| Harga tidak tersedia | Gunakan dataset lain atau hapus kriteria harga |
| Gender tidak seragam | Standarisasi menjadi male, female, unisex |
| Aroma terlalu banyak variasi | Kelompokkan ke fragrance family |
| Hasil rekomendasi kurang akurat | Perbaiki bobot dan rule |
| Data terlalu besar | Batasi hasil top-N dan optimasi preprocessing |
| TF-IDF tidak memahami konteks semantik dalam | Gunakan Sentence-BERT pada pengembangan lanjutan |

---

## 30. Evaluasi Sistem

Evaluasi dapat dilakukan dengan beberapa cara:

1. **Uji Kesesuaian Rule**
   - Memastikan rekomendasi sesuai dengan gender, umur, dan aroma.

2. **Uji Perhitungan Cosine Similarity**
   - Memastikan parfum dengan notes yang mirip mendapat skor lebih tinggi.

3. **Uji Perhitungan Fuzzy SAW**
   - Membandingkan hasil hitung manual dengan hasil sistem.

4. **Uji Black Box**
   - Menguji form input dan output rekomendasi.

5. **Uji User Acceptance**
   - Meminta beberapa pengguna mencoba sistem dan menilai apakah rekomendasi masuk akal.

6. **Uji Konsistensi**
   - Input yang sama harus menghasilkan ranking yang sama.

---

## 31. Success Metrics

Project dianggap berhasil jika:

| Indikator | Target |
|---|---|
| Dataset berhasil dibersihkan | Ya |
| Fitur teks aroma berhasil dibuat | Ya |
| TF-IDF dapat menghasilkan vektor teks | Ya |
| Cosine Similarity dapat menghasilkan skor aroma | Ya |
| Notebook dapat menghasilkan ranking | Ya |
| Aplikasi Flask dapat menerima input | Ya |
| Sistem dapat menghitung skor Fuzzy SAW | Ya |
| Sistem menampilkan minimal top 5 rekomendasi | Ya |
| Hasil rekomendasi memiliki alasan | Ya |
| Aplikasi dapat dijalankan secara lokal | Ya |

---

## 32. Minimum Viable Product

MVP project ini terdiri dari:

1. Dataset parfum bersih.
2. Notebook eksperimen hybrid recommendation.
3. Implementasi TF-IDF.
4. Implementasi Cosine Similarity.
5. Implementasi Fuzzy SAW.
6. Form input Flask.
7. Output top 5 parfum.
8. Skor aroma similarity.
9. Skor rekomendasi akhir.
10. Alasan rekomendasi.

---

## 33. Pengembangan Lanjutan

Fitur lanjutan yang dapat ditambahkan:

1. Login pengguna.
2. Riwayat rekomendasi.
3. Favorite parfum.
4. Perbandingan dua parfum.
5. Sentiment analysis dari review.
6. Content-based filtering yang lebih lengkap.
7. Sentence-BERT untuk similarity aroma yang lebih semantik.
8. Hybrid recommendation antara SPK dan machine learning lain.
9. Dashboard admin untuk menambah data parfum.
10. API rekomendasi parfum.
11. Deployment online.

---

## 34. Rekomendasi Judul Project

Beberapa pilihan judul:

1. **Sistem Rekomendasi Parfum Personal Menggunakan TF-IDF, Cosine Similarity, dan Fuzzy SAW Berbasis Web Flask**

2. **Hybrid Recommendation System untuk Rekomendasi Parfum Berdasarkan Jenis Kelamin, Umur, Aroma, Harga, dan Rating Menggunakan TF-IDF dan Fuzzy SAW**

3. **Aplikasi Rekomendasi Parfum Berbasis Machine Learning Klasik dan Sistem Pendukung Keputusan Menggunakan Flask**

4. **Sistem Pendukung Keputusan Rekomendasi Parfum Personal dengan Integrasi Cosine Similarity dan Fuzzy SAW**

---

## 35. Kesimpulan

Project ini bertujuan membangun sistem rekomendasi parfum personal berbasis hybrid antara Machine Learning klasik dan Sistem Pendukung Keputusan. Machine Learning digunakan melalui TF-IDF dan Cosine Similarity untuk menghitung kemiripan aroma antara input pengguna dan data parfum. Selanjutnya, metode Fuzzy SAW digunakan untuk menggabungkan skor kemiripan aroma dengan kriteria lain seperti gender, umur, rating, harga, dan fragrance family.

Tahap awal project dilakukan melalui notebook `.ipynb` untuk memahami dataset, membersihkan data, membuat fitur aroma, menghitung similarity, dan menguji Fuzzy SAW. Setelah metode berhasil diuji, sistem dikembangkan menjadi aplikasi web menggunakan Flask.

Keunggulan project ini adalah topiknya unik, modelnya ringan, mudah dijelaskan secara akademik, dan dapat dikembangkan menjadi aplikasi rekomendasi yang interaktif.
