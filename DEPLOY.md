# Panduan Deploy — Hugging Face Spaces (Docker)

Aplikasi dijalankan dalam container Docker di Hugging Face Spaces (gratis, RAM 16 GB),
versi **penuh SBERT + Fuzzy TOPSIS**. Repo git sudah disiapkan beserta Git LFS untuk
file model besar (`model.safetensors`, `perfume_embeddings.npy`, dll).

> File besar tidak perlu di-handle manual — `.gitattributes` sudah mengatur Git LFS.
> Folder `models/sbert_fuzzy_topsis/training_output/` (~520 MB) sengaja dikecualikan
> karena hanya sisa training dan tidak dipakai aplikasi.

---

## A. Push ke GitHub (opsional, untuk portofolio/source)

1. Buat repo kosong baru di <https://github.com/new> — **jangan** centang "Add a README".
   Misal nama repo: `perfume-dss`.

2. Hubungkan dan push (ganti `Fakhri428` bila perlu):

   ```bash
   git remote add origin https://github.com/Fakhri428/perfume-dss.git
   git push -u origin main
   ```

   Saat diminta password, gunakan **Personal Access Token** GitHub
   (Settings → Developer settings → Tokens), bukan password akun.

   > Catatan: GitHub LFS gratis 1 GB storage / 1 GB bandwidth per bulan.
   > Total file LFS project ini ~150 MB, jadi masih aman.

---

## B. Deploy ke Hugging Face Spaces (utama)

1. Login HF CLI (butuh token **write** dari <https://huggingface.co/settings/tokens>):

   ```bash
   huggingface-cli login
   ```

2. Buat Space baru bertipe Docker (ganti `<user>` dengan username HF Anda):

   ```bash
   huggingface-cli repo create perfume-dss --type space --space_sdk docker
   ```

   Atau via web: <https://huggingface.co/new-space> → pilih **Docker** sebagai SDK.

3. Tambahkan remote Space dan push:

   ```bash
   git remote add space https://huggingface.co/spaces/<user>/perfume-dss
   git push space main
   ```

   Saat diminta password, paste **token HF** (write access).

4. HF otomatis mem-build `Dockerfile`. Tunggu status **Running**, lalu buka:

   ```text
   https://<user>-perfume-dss.hf.space
   ```

   Build pertama agak lama (install torch CPU + dependency). Setelah itu cepat.

---

## Catatan penting

- **Port**: container listen di `7860` (sesuai `app_port` di frontmatter `README.md`).
- **Log rekomendasi**: file `data/recommendation_logs.json` bersifat *ephemeral* —
  akan ter-reset saat Space rebuild/restart. Untuk demo tidak masalah. Jika perlu
  permanen, sambungkan ke database eksternal (mis. Supabase free).
- **Update aplikasi**: cukup `git commit` lalu `git push space main` (dan/atau
  `git push origin main`); HF rebuild otomatis.
- **Mode fallback**: bila suatu saat artefak SBERT dihapus, aplikasi otomatis turun
  ke TF-IDF + SAW (lihat `app/recommender.py`) sehingga tetap jalan.
