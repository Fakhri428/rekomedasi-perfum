from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.spk import UserPreference, age_group_label

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "data"
LOG_PATH = LOG_DIR / "recommendation_logs.json"

# Batasi jumlah log yang disimpan agar file tidak tumbuh tanpa batas.
MAX_LOGS = 100

_LOCK = threading.Lock()

_MONTHS_ID = [
    "Jan", "Feb", "Mar", "Apr", "Mei", "Jun",
    "Jul", "Agu", "Sep", "Okt", "Nov", "Des",
]


def _to_jsonable(value: Any) -> Any:
    """Konversi tipe numpy/pandas agar bisa diserialisasi ke JSON."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _format_display(moment: datetime) -> str:
    return f"{moment.day:02d} {_MONTHS_ID[moment.month - 1]} {moment.year}, {moment:%H:%M}"


def _read_all() -> list[dict[str, Any]]:
    if not LOG_PATH.exists():
        return []
    try:
        with LOG_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return data
        return []
    except (json.JSONDecodeError, OSError):
        return []


def _write_all(logs: list[dict[str, Any]]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("w", encoding="utf-8") as handle:
        json.dump(logs, handle, ensure_ascii=False, indent=2)


def save_log(
    preference: UserPreference,
    results: list[dict[str, Any]],
    model_source: str = "",
) -> str:
    """Simpan satu sesi rekomendasi ke log dan kembalikan id-nya."""
    moment = datetime.now()
    entry = {
        "id": uuid.uuid4().hex,
        "created_at": moment.isoformat(timespec="seconds"),
        "created_display": _format_display(moment),
        "model_source": model_source,
        "age_group": age_group_label(preference.age),
        "preference": {
            "gender": preference.gender,
            "age": preference.age,
            "fragrance_family": preference.fragrance_family,
            "aroma_keywords": preference.aroma_keywords,
            "budget": preference.budget,
            "minimal_rating": preference.minimal_rating,
            "usage": preference.usage,
            "top_n": preference.top_n,
        },
        "result_count": len(results),
        "results": _to_jsonable(results),
    }

    with _LOCK:
        logs = _read_all()
        logs.insert(0, entry)
        del logs[MAX_LOGS:]
        _write_all(logs)

    return entry["id"]


def list_logs() -> list[dict[str, Any]]:
    """Daftar ringkas semua log, terbaru di atas."""
    with _LOCK:
        return _read_all()


def get_log(log_id: str) -> dict[str, Any] | None:
    with _LOCK:
        for entry in _read_all():
            if entry.get("id") == log_id:
                return entry
    return None


def delete_log(log_id: str) -> bool:
    with _LOCK:
        logs = _read_all()
        remaining = [entry for entry in logs if entry.get("id") != log_id]
        if len(remaining) == len(logs):
            return False
        _write_all(remaining)
    return True


def clear_logs() -> None:
    with _LOCK:
        _write_all([])
