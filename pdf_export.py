"""
pdf_export.py
-------------
Typesetting layer: converts Markdown-formatted study notes and a
plain-prose summary into a beautifully laid-out PDF using ReportLab's
Platypus (flowable) engine.

Markdown subset supported
  **bold**        → <b>bold</b>
  *italic*        → <i>italic</i>
  `code`          → <font name="Courier">code</font>
  ### Heading     → Heading1 style
  ## Heading      → (treated as Heading1 — top-level section)
  - bullet        → ListItem bullet
  1. numbered     → ListItem numbered
  blank line      → Spacer

The generated PDF is saved to *generated_pdfs/* alongside the project
root and the absolute file path is returned to the caller.
"""

import os
import re
import uuid
import logging
from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    HRFlowable,
    ListFlowable,
    ListItem,
    KeepTogether,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
_OUTPUT_DIR = Path("generated_pdfs")


# ---------------------------------------------------------------------------
# Colour palette  (high-contrast academic)
# ---------------------------------------------------------------------------
class _Palette:
    INK         = colors.HexColor("#1A1A2E")   # near-black for body text
    HEADING     = colors.HexColor("#16213E")   # deep navy for headings
    ACCENT      = colors.HexColor("#0F3460")   # medium navy for sub-headings
    RULE        = colors.HexColor("#E94560")   # crimson rule / accent line
    META        = colors.HexColor("#4A4A6A")   # muted slate for meta / captions
    PAGE_BG     = colors.white
    BULLET_CLR  = colors.HexColor("#E94560")   # bullet markers


# ---------------------------------------------------------------------------
# Style sheet factory
# ---------------------------------------------------------------------------

def _build_stylesheet():
    base = getSampleStyleSheet()

    styles: dict[str, ParagraphStyle] = {}

    styles["Title"] = ParagraphStyle(
        "Title",
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=28,
        textColor=_Palette.HEADING,
        alignment=TA_CENTER,
        spaceAfter=6,
    )

    styles["Meta"] = ParagraphStyle(
        "Meta",
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=13,
        textColor=_Palette.META,
        alignment=TA_CENTER,
        spaceAfter=14,
    )

    styles["H1"] = ParagraphStyle(
        "H1",
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=20,
        textColor=_Palette.HEADING,
        spaceBefore=16,
        spaceAfter=4,
    )

    styles["H2"] = ParagraphStyle(
        "H2",
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=_Palette.ACCENT,
        spaceBefore=12,
        spaceAfter=3,
    )

    styles["SectionLabel"] = ParagraphStyle(
        "SectionLabel",
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=_Palette.RULE,
        spaceBefore=20,
        spaceAfter=2,
        letterSpacing=1.5,
    )

    styles["Body"] = ParagraphStyle(
        "Body",
        fontName="Helvetica",
        fontSize=10,
        leading=15,
        textColor=_Palette.INK,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )

    styles["BulletText"] = ParagraphStyle(
        "BulletText",
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=_Palette.INK,
        leftIndent=0,
    )

    styles["Code"] = ParagraphStyle(
        "Code",
        fontName="Courier",
        fontSize=9,
        leading=13,
        textColor=_Palette.ACCENT,
        backColor=colors.HexColor("#F0F0F5"),
        spaceAfter=4,
    )

    return styles


# ---------------------------------------------------------------------------
# Markdown → ReportLab-XML conversion helpers
# ---------------------------------------------------------------------------

def _md_inline_to_rl(text: str) -> str:
    """
    Convert inline Markdown markup inside *text* to ReportLab XML tags.
    Order matters: bold before italic to handle ***text*** correctly.
    """
    # Bold: **text** or __text__
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)
    # Italic: *text* or _text_  (not already consumed by bold)
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    text = re.sub(r"_(.+?)_", r"<i>\1</i>", text)
    # Inline code: `code`
    text = re.sub(
        r"`(.+?)`",
        r'<font name="Courier" color="#0F3460">\1</font>',
        text,
    )
    return text


def _escape_rl_special(text: str) -> str:
    """
    Escape ampersands and angle brackets that are NOT part of already-
    inserted ReportLab XML tags so that RML parsing doesn't choke.
    """
    # Protect existing tags from double-escaping
    placeholder_map: dict[str, str] = {}

    def stash(m: re.Match) -> str:
        key = f"\x00TAG{len(placeholder_map)}\x00"
        placeholder_map[key] = m.group(0)
        return key

    protected = re.sub(r"<[^>]+>", stash, text)

    protected = protected.replace("&", "&amp;")
    # Restore
    for key, original in placeholder_map.items():
        protected = protected.replace(key, original)

    return protected


def _process_line(line: str, styles: dict) -> tuple[str, object | None]:
    """
    Classify a single Markdown line and return (type_tag, flowable_or_None).
    Returns:
      ("h1", Paragraph)  for ## / ### headings
      ("bullet", str)    raw text for list accumulation
      ("numbered", str)  raw text for numbered list accumulation
      ("body", Paragraph) for regular prose
      ("blank", None)    for empty lines
    """
    stripped = line.strip()

    if not stripped:
        return ("blank", None)

    # Heading level 1 (##  or ###)
    if stripped.startswith("### "):
        content = _md_inline_to_rl(_escape_rl_special(stripped[4:].strip()))
        return ("h1", Paragraph(content, styles["H1"]))

    if stripped.startswith("## "):
        content = _md_inline_to_rl(_escape_rl_special(stripped[3:].strip()))
        return ("h1", Paragraph(content, styles["H1"]))

    if stripped.startswith("# "):
        content = _md_inline_to_rl(_escape_rl_special(stripped[2:].strip()))
        return ("h1", Paragraph(content, styles["H1"]))

    # Bullet list item
    if re.match(r"^[-*+]\s+", stripped):
        content = re.sub(r"^[-*+]\s+", "", stripped)
        return ("bullet", _md_inline_to_rl(_escape_rl_special(content)))

    # Numbered list item
    if re.match(r"^\d+\.\s+", stripped):
        content = re.sub(r"^\d+\.\s+", "", stripped)
        return ("numbered", _md_inline_to_rl(_escape_rl_special(content)))

    # Plain body paragraph
    content = _md_inline_to_rl(_escape_rl_special(stripped))
    return ("body", Paragraph(content, styles["Body"]))


def _flush_list(
    items: list[str],
    list_type: str,
    styles: dict,
) -> ListFlowable:
    """Build a ReportLab ListFlowable from accumulated list items."""
    bullet_style = "bullet" if list_type == "bullet" else "1"
    list_items = [
        ListItem(
            Paragraph(item, styles["BulletText"]),
            leftIndent=18,
            bulletColor=_Palette.BULLET_CLR,
        )
        for item in items
    ]
    return ListFlowable(
        list_items,
        bulletType=bullet_style,
        start=None if list_type == "bullet" else 1,
        leftIndent=14,
        spaceBefore=4,
        spaceAfter=6,
    )


def _markdown_to_flowables(
    md_text: str,
    styles: dict,
    section_label: str | None = None,
) -> list:
    """
    Parse *md_text* (Markdown string) into a list of ReportLab flowables.
    Optionally prepend a coloured section label above the content.
    """
    flowables: list = []

    if section_label:
        flowables.append(
            Paragraph(section_label.upper(), styles["SectionLabel"])
        )
        flowables.append(
            HRFlowable(
                width="100%",
                thickness=1.5,
                color=_Palette.RULE,
                spaceAfter=6,
            )
        )

    lines = md_text.splitlines()
    pending_list_items: list[str] = []
    pending_list_type: str | None = None

    def flush_pending() -> None:
        nonlocal pending_list_items, pending_list_type
        if pending_list_items:
            flowables.append(
                _flush_list(pending_list_items, pending_list_type, styles)
            )
            pending_list_items = []
            pending_list_type = None

    for line in lines:
        tag, element = _process_line(line, styles)

        if tag in ("bullet", "numbered"):
            if pending_list_type and pending_list_type != tag:
                flush_pending()
            pending_list_type = tag
            pending_list_items.append(element)  # element is str here
        else:
            flush_pending()
            if tag == "blank":
                flowables.append(Spacer(1, 0.2 * cm))
            elif tag == "h1":
                flowables.append(element)
            elif tag == "body":
                flowables.append(element)

    flush_pending()
    return flowables


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_to_pdf(
    notes: str,
    summary: str,
    transcript: str,
    filename_prefix: str = "lecture_notes",
) -> str:
    """
    Render *notes*, *summary*, and *transcript* into a typeset PDF.

    Parameters
    ----------
    notes : str
        Markdown-formatted structured study notes.
    summary : str
        Plain-prose executive summary.
    transcript : str
        Raw verbatim transcript.
    filename_prefix : str
        Stem used when building the output filename.

    Returns
    -------
    str
        Absolute path of the generated PDF file.

    Raises
    ------
    IOError
        If the output directory cannot be created or the file cannot
        be written.
    """
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    unique_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}_{unique_id}.pdf"
    output_path = str((_OUTPUT_DIR / filename).resolve())

    styles = _build_stylesheet()

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2.5 * cm,
        rightMargin=2.5 * cm,
        topMargin=2.5 * cm,
        bottomMargin=2.2 * cm,
        title="Lecture Study Notes",
        author="Lecture Notes Converter",
    )

    story: list = []

    # ── Cover / Title block ────────────────────────────────────────────
    story.append(Spacer(1, 0.8 * cm))
    story.append(Paragraph("Lecture Study Notes", styles["Title"]))
    story.append(
        Paragraph(
            f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}",
            styles["Meta"],
        )
    )
    story.append(
        HRFlowable(
            width="100%",
            thickness=3,
            color=_Palette.RULE,
            spaceAfter=20,
        )
    )

    # ── Executive Summary ──────────────────────────────────────────────
    story.extend(
        _markdown_to_flowables(summary, styles, section_label="Executive Summary")
    )
    story.append(Spacer(1, 0.6 * cm))

    # ── Structured Study Notes ─────────────────────────────────────────
    story.extend(
        _markdown_to_flowables(notes, styles, section_label="Structured Study Notes")
    )
    story.append(Spacer(1, 0.6 * cm))

    # ── Raw Transcript ─────────────────────────────────────────────────
    story.append(
        Paragraph("RAW TRANSCRIPT", styles["SectionLabel"])
    )
    story.append(
        HRFlowable(
            width="100%",
            thickness=1.5,
            color=_Palette.RULE,
            spaceAfter=6,
        )
    )

    transcript_style = ParagraphStyle(
        "TranscriptBody",
        parent=styles["Body"],
        fontSize=9,
        leading=13,
        textColor=_Palette.META,
        alignment=TA_LEFT,
    )

    # Break transcript into ~80-char visual chunks for legibility
    for chunk in _chunk_transcript(transcript, max_chars=600):
        story.append(Paragraph(_escape_rl_special(chunk), transcript_style))
        story.append(Spacer(1, 0.15 * cm))

    # ── Build PDF ──────────────────────────────────────────────────────
    try:
        doc.build(story)
    except Exception as exc:
        raise IOError(f"ReportLab failed to build PDF: {exc}") from exc

    logger.info("PDF exported to: %s", output_path)
    return output_path


def _chunk_transcript(text: str, max_chars: int = 600) -> list[str]:
    """
    Split *text* into chunks of at most *max_chars* characters,
    breaking on sentence boundaries where possible.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    while text:
        if len(text) <= max_chars:
            chunks.append(text)
            break
        split_at = text.rfind(". ", 0, max_chars)
        if split_at == -1:
            split_at = text.rfind(" ", 0, max_chars)
        if split_at == -1:
            split_at = max_chars
        else:
            split_at += 1  # include the period
        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()

    return chunks
