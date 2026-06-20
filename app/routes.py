from __future__ import annotations

from flask import (
    Blueprint,
    Response,
    abort,
    redirect,
    render_template,
    request,
    url_for,
)

from app import log_store
from app.pdf_export import build_log_pdf
from app.recommender import get_recommender, preference_from_mapping
from app.spk import age_group_label


main_bp = Blueprint("main", __name__)


@main_bp.route("/", methods=["GET"])
def index():
    recommender = get_recommender()
    return render_template(
        "index.html",
        families=recommender.fragrance_families(),
        total_items=recommender.total_items,
    )


@main_bp.route("/recommend", methods=["POST"])
def recommend():
    recommender = get_recommender()
    preference = preference_from_mapping(request.form)
    results = recommender.recommend(preference)
    log_id = log_store.save_log(preference, results, recommender.model_source)
    return render_template(
        "result.html",
        preference=preference,
        age_group=age_group_label(preference.age),
        results=results,
        families=recommender.fragrance_families(),
        log_id=log_id,
    )


@main_bp.route("/perfume/<int:perfume_id>", methods=["GET"])
def detail(perfume_id: int):
    recommender = get_recommender()
    perfume = recommender.get_perfume(perfume_id)
    if perfume is None:
        abort(404)
    return render_template("detail.html", perfume=perfume)


@main_bp.route("/log", methods=["GET"])
def log_list():
    logs = log_store.list_logs()
    return render_template("log.html", logs=logs)


@main_bp.route("/log/<log_id>", methods=["GET"])
def log_detail(log_id: str):
    entry = log_store.get_log(log_id)
    if entry is None:
        abort(404)
    return render_template("log_detail.html", entry=entry)


@main_bp.route("/log/<log_id>/pdf", methods=["GET"])
def log_pdf(log_id: str):
    entry = log_store.get_log(log_id)
    if entry is None:
        abort(404)
    pdf_bytes = build_log_pdf(entry)
    filename = f"rekomendasi-{entry.get('created_at', log_id)[:10]}-{log_id[:8]}.pdf"
    return Response(
        pdf_bytes,
        mimetype="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@main_bp.route("/log/<log_id>/delete", methods=["POST"])
def log_delete(log_id: str):
    log_store.delete_log(log_id)
    return redirect(url_for("main.log_list"))


@main_bp.route("/log/clear", methods=["POST"])
def log_clear():
    log_store.clear_logs()
    return redirect(url_for("main.log_list"))
