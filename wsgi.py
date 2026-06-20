"""Entry point WSGI untuk server produksi (gunicorn wsgi:app).

Catatan: `import app` mengacu ke package `app/` (bukan file `app.py`),
sehingga entrypoint khusus ini menghindari ambiguitas saat deploy.
"""
from app import create_app

app = create_app()
