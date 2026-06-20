from __future__ import annotations

import io
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ACCENT = colors.HexColor("#7f1d3a")
INK = colors.HexColor("#171717")
MUTED = colors.HexColor("#73766f")
LINE = colors.HexColor("#deded7")
SOFT_BG = colors.HexColor("#fbfaf7")


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "DssTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            textColor=INK,
            spaceAfter=2,
            alignment=TA_LEFT,
        ),
        "eyebrow": ParagraphStyle(
            "DssEyebrow",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8,
            textColor=ACCENT,
            spaceAfter=4,
        ),
        "meta": ParagraphStyle(
            "DssMeta",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9,
            textColor=MUTED,
            spaceAfter=1,
        ),
        "section": ParagraphStyle(
            "DssSection",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            textColor=INK,
            spaceBefore=12,
            spaceAfter=6,
        ),
        "cell": ParagraphStyle(
            "DssCell",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=8.5,
            textColor=INK,
            leading=11,
        ),
        "cell_head": ParagraphStyle(
            "DssCellHead",
            parent=base["Normal"],
            fontName="Helvetica-Bold",
            fontSize=8.5,
            textColor=colors.white,
            leading=11,
        ),
        "footer": ParagraphStyle(
            "DssFooter",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            textColor=MUTED,
        ),
    }


def _fmt_float(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _preference_table(entry: dict[str, Any], styles: dict[str, ParagraphStyle]) -> Table:
    pref = entry.get("preference", {})
    budget = pref.get("budget", 0) or 0
    rows = [
        ("Jenis kelamin", pref.get("gender", "-")),
        ("Umur", f"{pref.get('age', '-')} ({entry.get('age_group', '-')})"),
        ("Fragrance family", pref.get("fragrance_family") or "Bebas"),
        ("Aroma / notes", pref.get("aroma_keywords") or "-"),
        ("Pemakaian", pref.get("usage", "-")),
        ("Budget", f"{float(budget):,.0f}"),
        ("Minimal rating", _fmt_float(pref.get("minimal_rating", 0), 1)),
        ("Jumlah hasil", str(pref.get("top_n", "-"))),
    ]
    data = [
        [Paragraph(label, styles["meta"]), Paragraph(str(value), styles["cell"])]
        for label, value in rows
    ]
    table = Table(data, colWidths=[42 * mm, 122 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), SOFT_BG),
                ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def _results_table(entry: dict[str, Any], styles: dict[str, ParagraphStyle]) -> Table:
    header = ["#", "Parfum", "Brand", "Family", "Cosine", "Skor", "Rating", "Harga"]
    data = [[Paragraph(text, styles["cell_head"]) for text in header]]

    for result in entry.get("results", []):
        data.append(
            [
                Paragraph(str(result.get("rank", "")), styles["cell"]),
                Paragraph(str(result.get("name", "-")), styles["cell"]),
                Paragraph(str(result.get("brand", "-")), styles["cell"]),
                Paragraph(str(result.get("fragrance_family", "-")), styles["cell"]),
                Paragraph(_fmt_float(result.get("aroma_similarity")), styles["cell"]),
                Paragraph(_fmt_float(result.get("final_score")), styles["cell"]),
                Paragraph(str(result.get("display_rating", "-")), styles["cell"]),
                Paragraph(str(result.get("display_price", "-")), styles["cell"]),
            ]
        )

    table = Table(
        data,
        colWidths=[8 * mm, 46 * mm, 30 * mm, 28 * mm, 14 * mm, 14 * mm, 14 * mm, 18 * mm],
        repeatRows=1,
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("ALIGN", (4, 0), (-1, -1), "CENTER"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, SOFT_BG]),
                ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def build_log_pdf(entry: dict[str, Any]) -> bytes:
    """Render satu entri log rekomendasi menjadi dokumen PDF (bytes)."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        title="Log Rekomendasi Parfum",
    )

    styles = _styles()
    story: list[Any] = []

    story.append(Paragraph("PERFUMEDSS &middot; SBERT + FUZZY TOPSIS", styles["eyebrow"]))
    story.append(Paragraph("Log Rekomendasi Parfum", styles["title"]))
    story.append(Paragraph(f"Tanggal: {entry.get('created_display', '-')}", styles["meta"]))
    if entry.get("model_source"):
        story.append(Paragraph(f"Model: {entry['model_source']}", styles["meta"]))
    story.append(Spacer(1, 6))

    story.append(Paragraph("Preferensi Pengguna", styles["section"]))
    story.append(_preference_table(entry, styles))

    story.append(Paragraph("Hasil Rekomendasi", styles["section"]))
    if entry.get("results"):
        story.append(_results_table(entry, styles))
    else:
        story.append(Paragraph("Tidak ada hasil rekomendasi.", styles["cell"]))

    story.append(Spacer(1, 14))
    story.append(
        Paragraph(
            "Dokumen ini dihasilkan otomatis oleh PerfumeDSS sebagai catatan rekomendasi.",
            styles["footer"],
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer.read()
