"""
pdf_generator.py
─────────────────────────────────────────────────────────────────
Generates a professional analysis PDF using ReportLab.
The output looks like a real WHO-style situation report.
"""

import io
from datetime import datetime
from typing import Dict

from reportlab.lib.pagesizes  import A4
from reportlab.lib.units       import mm
from reportlab.lib             import colors
from reportlab.lib.styles      import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums       import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus        import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether,
)
from reportlab.platypus        import PageBreak

from rag_engine import analyse_document

from extractor import extract_full_text , get_page_count 


# ── Color palette ──────────────────────────────────────────────────────────────
BRAND_DARK   = colors.HexColor("#0A0F1E")
BRAND_PURPLE = colors.HexColor("#7C3AED")
BRAND_LIGHT  = colors.HexColor("#EDE9FE")
RISK_COLORS  = {
    "critical": colors.HexColor("#FF1744"),
    "high":     colors.HexColor("#FF6D00"),
    "medium":   colors.HexColor("#FFD600"),
    "low":      colors.HexColor("#00C853"),
}
GREY_LIGHT   = colors.HexColor("#F1F5F9")
GREY_MID     = colors.HexColor("#94A3B8")
GREY_DARK    = colors.HexColor("#334155")
WHITE        = colors.white
BLACK        = colors.black


def _risk_color(level: str) -> colors.Color:
    return RISK_COLORS.get(level.lower(), GREY_MID)


# ── Style helpers ──────────────────────────────────────────────────────────────
def _styles():
    base = getSampleStyleSheet()
    custom = {
        "title": ParagraphStyle(
            "title",
            fontSize=22, fontName="Helvetica-Bold",
            textColor=WHITE, alignment=TA_LEFT,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            fontSize=10, fontName="Helvetica",
            textColor=colors.HexColor("#A78BFA"), alignment=TA_LEFT,
            spaceAfter=2,
        ),
        "section": ParagraphStyle(
            "section",
            fontSize=11, fontName="Helvetica-Bold",
            textColor=BRAND_PURPLE, spaceBefore=14, spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            fontSize=9, fontName="Helvetica",
            textColor=GREY_DARK, spaceAfter=4, leading=14,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            fontSize=9, fontName="Helvetica",
            textColor=GREY_DARK, spaceAfter=3, leading=13,
            leftIndent=12, bulletIndent=0,
        ),
        "caption": ParagraphStyle(
            "caption",
            fontSize=8, fontName="Helvetica-Oblique",
            textColor=GREY_MID, alignment=TA_RIGHT,
        ),
        "risk_badge": ParagraphStyle(
            "risk_badge",
            fontSize=13, fontName="Helvetica-Bold",
            textColor=WHITE, alignment=TA_CENTER,
        ),
    }
    return custom


# ── Header banner ──────────────────────────────────────────────────────────────
def _header_table(analysis: Dict, doc_id: str, styles: dict):
    """Dark header banner with title + metadata."""
    generated = datetime.now().strftime("%d %B %Y, %H:%M")
    risk       = analysis.get("risk_level", "unknown").upper()
    risk_color = _risk_color(analysis.get("risk_level", ""))

    title_cell = [
        Paragraph("SHASTHO AI", styles["subtitle"]),
        Paragraph("Health Report Analysis", styles["title"]),
        Paragraph(f"Document ID: {doc_id}  ·  Generated: {generated}", styles["subtitle"]),
    ]

    badge_cell = [
        Paragraph("OVERALL RISK", ParagraphStyle("rb_label", fontSize=8,
            fontName="Helvetica-Bold", textColor=colors.HexColor("#A78BFA"),
            alignment=TA_CENTER, spaceAfter=4)),
        Paragraph(risk, styles["risk_badge"]),
    ]

    tbl = Table(
        [[title_cell, badge_cell]],
        colWidths=[130 * mm, 40 * mm],
    )
    tbl.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, -1), BRAND_DARK),
        ("BACKGROUND",  (1, 0), (1, 0),   risk_color),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (0, 0),   10),
        ("RIGHTPADDING", (1, 0), (1, 0),  8),
        ("TOPPADDING",  (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 12),
        ("ROUNDEDCORNERS", [4]),
    ]))
    return tbl


# ── Stat row ───────────────────────────────────────────────────────────────────
def _stats_table(analysis: Dict):
    stats = [
        ("Pages",       analysis.get("pages",           "—")),
        ("Text Chunks", analysis.get("chunks",          "—")),
        ("Diseases",    analysis.get("diseases_found",  "—")),
        ("Districts",   analysis.get("districts_found", "—")),
    ]

    cells = []
    for label, value in stats:
        cells.append([
            Paragraph(str(value), ParagraphStyle("sv", fontSize=18,
                fontName="Helvetica-Bold", textColor=BRAND_PURPLE, alignment=TA_CENTER)),
            Paragraph(label, ParagraphStyle("sl", fontSize=8,
                fontName="Helvetica", textColor=GREY_MID, alignment=TA_CENTER)),
        ])

    tbl = Table([cells], colWidths=[42 * mm] * 4)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), GREY_LIGHT),
        ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING",   (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 8),
        ("ROUNDEDCORNERS", [4]),
    ]))
    return tbl


# ── Bullet list ────────────────────────────────────────────────────────────────
def _bullet_list(items, styles, icon="▸", color=BRAND_PURPLE):
    flowables = []
    for item in items:
        p = Paragraph(
            f'<font color="{color.hexval() if hasattr(color,"hexval") else "#7C3AED"}">{icon}</font>  {item}',
            styles["bullet"],
        )
        flowables.append(p)
    return flowables


# ── Main generator ─────────────────────────────────────────────────────────────
def generate_report_pdf(analysis: Dict, doc_id: str, original_filename: str = "") -> bytes:
    """
    Build a professional PDF analysis report.

    Args:
        analysis:          The dict returned by rag_engine.analyse_document()
        doc_id:            Document identifier string
        original_filename: Optional original filename for display

    Returns:
        PDF as bytes (ready to stream to client).
    """
    buffer = io.BytesIO()
    styles = _styles()

    W, H   = A4
    margin = 18 * mm

    doc = SimpleDocTemplate(
        buffer,
        pagesize      = A4,
        leftMargin    = margin,
        rightMargin   = margin,
        topMargin     = margin,
        bottomMargin  = margin,
        title         = "Shastho AI — Report Analysis",
        author        = "Shastho AI Engine",
    )

    story = []

    # ── Header ──
    story.append(_header_table(analysis, doc_id, styles))
    story.append(Spacer(1, 6 * mm))

    # ── Original file name (if provided) ──
    if original_filename:
        story.append(Paragraph(
            f"Source file: <b>{original_filename}</b>",
            styles["caption"],
        ))
        story.append(Spacer(1, 4 * mm))

    # ── Stats row ──
    story.append(_stats_table(analysis))
    story.append(Spacer(1, 6 * mm))

    # ── Divider ──
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E2E8F0")))
    story.append(Spacer(1, 4 * mm))

    # ── Executive summary ──
    if analysis.get("summary"):
        story.append(Paragraph("Executive Summary", styles["section"]))
        story.append(Paragraph(analysis["summary"], styles["body"]))
        story.append(Spacer(1, 3 * mm))

    # ── Key findings ──
    findings = analysis.get("key_findings", [])
    if findings:
        story.append(KeepTogether([
            Paragraph("Key Findings", styles["section"]),
            *_bullet_list(findings, styles, icon="▸", color=BRAND_PURPLE),
        ]))
        story.append(Spacer(1, 3 * mm))

    # ── Recommendations ──
    recs = analysis.get("recommendations", [])
    if recs:
        story.append(KeepTogether([
            Paragraph("Recommendations", styles["section"]),
            *_bullet_list(recs, styles, icon="✓", color=colors.HexColor("#00C853")),
        ]))
        story.append(Spacer(1, 3 * mm))

    # ── Risk assessment box ──
    risk_level = analysis.get("risk_level", "unknown")
    risk_color = _risk_color(risk_level)
    risk_desc  = {
        "critical": "Immediate intervention required. Deploy emergency resources now.",
        "high":     "Urgent action recommended. Monitor closely and allocate resources.",
        "medium":   "Elevated risk. Increase surveillance and prepare response plans.",
        "low":      "Situation under control. Maintain standard monitoring protocols.",
    }.get(risk_level, "Risk level undetermined.")

    risk_tbl = Table(
        [[
            Paragraph(f"RISK LEVEL: {risk_level.upper()}", ParagraphStyle(
                "rh", fontSize=11, fontName="Helvetica-Bold", textColor=WHITE,
            )),
            Paragraph(risk_desc, ParagraphStyle(
                "rd", fontSize=9, fontName="Helvetica", textColor=WHITE, leading=13,
            )),
        ]],
        colWidths=[45 * mm, 120 * mm],
    )
    risk_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, -1), risk_color),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",   (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 10),
        ("LEFTPADDING",  (0, 0), (0, 0),   10),
        ("LEFTPADDING",  (1, 0), (1, 0),   10),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(risk_tbl)
    story.append(Spacer(1, 6 * mm))

    # ── Footer note ──
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E2E8F0")))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "This report was generated by Shastho AI using RAG (Retrieval-Augmented Generation). "
        "All findings are derived from the uploaded source document. "
        "This is an AI-generated analysis — always verify with qualified health professionals.",
        ParagraphStyle("footer", fontSize=7, fontName="Helvetica-Oblique",
                       textColor=GREY_MID, leading=11),
    ))

    # ── Build ──
    doc.build(story)
    return buffer.getvalue()


stordata  = extract_full_text('./pdf/T1.2.pdf')
pagecount = get_page_count('./pdf/T1.2.pdf')


resultstor = analyse_document(stordata, pagecount, 6)
# print(resultstor)

pdf_bytes=generate_report_pdf(resultstor,"91101",'T1.2.pdf')

'''


# Save to file
with open("report.pdf", "wb") as f:
    f.write(pdf_bytes)

# Auto-open in your default PDF viewer
import os
os.startfile("report.pdf")  # Windows
# os.system("open report.pdf")    # Mac
# os.system("xdg-open report.pdf") # Linux
'''