# Dockerfile untuk deploy ke Hugging Face Spaces (SDK: docker)
# Menjalankan versi penuh SBERT + Fuzzy TOPSIS.
FROM python:3.11-slim

# HF Spaces menjalankan container sebagai user dengan UID 1000.
RUN useradd -m -u 1000 user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PORT=7860

WORKDIR /app

# Install torch versi CPU dulu (jauh lebih kecil daripada build CUDA),
# lalu sisa dependency. sentence-transformers tidak akan menarik ulang torch.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Salin seluruh project (lihat .dockerignore untuk yang dikecualikan).
COPY . .

# Folder log rekomendasi harus bisa ditulis oleh user runtime.
RUN mkdir -p /app/data && chown -R user:user /app

USER user

EXPOSE 7860

# 1 worker agar hemat RAM (model SBERT dimuat sekali per worker).
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "180", "app:app"]
