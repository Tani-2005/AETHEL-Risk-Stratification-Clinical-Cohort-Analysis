"""
export_utils.py
===============
Handles file exports to various formats.
- Figures: SVG, PDF, PNG (300 DPI)
- Tables: CSV, Excel, LaTeX, Markdown
- Reports: HTML, Markdown, PDF (via ReportLab)
"""
from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ReportLab imports
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """
    Canvas to handle two-pass page numbering ('Page X of Y')
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_number(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#7f8c8d"))
        
        # Header (on all pages except the first)
        if self._pageNumber > 1:
            self.drawString(54, 750, "AETHEL Clinical AI Evaluation & Trustworthiness Report")
            self.setStrokeColor(colors.HexColor("#bdc3c7"))
            self.setLineWidth(0.5)
            self.line(54, 742, 558, 742)
            
        # Footer
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 40, page_str)
        self.drawString(54, 40, "CONFIDENTIAL — FOR ACADEMIC/RESEARCH USE ONLY")
        self.setStrokeColor(colors.HexColor("#bdc3c7"))
        self.setLineWidth(0.5)
        self.line(54, 52, 558, 52)
        
        self.restoreState()


def export_figure(fig: plt.Figure, output_dir: Path, filename_base: str) -> dict[str, Path]:
    """
    Exports a Matplotlib figure in PNG, SVG, and PDF.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    
    # Save as PNG
    png_path = output_dir / f"{filename_base}.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    paths["png"] = png_path
    
    # Save as SVG (vector)
    svg_path = output_dir / f"{filename_base}.svg"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    paths["svg"] = svg_path
    
    # Save as PDF (vector)
    pdf_path = output_dir / f"{filename_base}.pdf"
    fig.savefig(pdf_path, format="pdf", bbox_inches="tight")
    paths["pdf"] = pdf_path
    
    return paths


def export_table(df: pd.DataFrame, output_dir: Path, filename_base: str) -> dict[str, Path]:
    """
    Exports a DataFrame in CSV, Excel, LaTeX, and Markdown.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    
    # Save CSV
    csv_path = output_dir / f"{filename_base}.csv"
    df.to_csv(csv_path, index=False)
    paths["csv"] = csv_path
    
    # Save Excel
    xlsx_path = output_dir / f"{filename_base}.xlsx"
    df.to_excel(xlsx_path, index=False, engine="openpyxl")
    paths["xlsx"] = xlsx_path
    
    # Save LaTeX
    tex_path = output_dir / f"{filename_base}.tex"
    df.to_latex(tex_path, index=False, column_format="l" + "c"*(len(df.columns)-1))
    paths["tex"] = tex_path
    
    # Save Markdown
    md_path = output_dir / f"{filename_base}.md"
    df.to_markdown(md_path, index=False)
    paths["md"] = md_path
    
    return paths


def build_pdf_report(
    title: str,
    metadata: dict,
    content_list: list[dict],
    output_path: Path
) -> Path:
    """
    Builds a professional PDF document using ReportLab.
    
    content_list contains elements:
    - {"type": "h1", "text": "Heading 1"}
    - {"type": "h2", "text": "Heading 2"}
    - {"type": "p", "text": "Paragraph text"}
    - {"type": "table", "data": pd.DataFrame or list of lists}
    - {"type": "image", "path": Path}
    - {"type": "page_break"}
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Styles matching our visual layout
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#2c3e50"),
        alignment=0, # Left-aligned
        spaceAfter=20
    )
    
    h1_style = ParagraphStyle(
        "H1Style",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1f77b4"), # Steel Blue
        spaceBefore=15,
        spaceAfter=8,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        "H2Style",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#2c3e50"),
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#34495e"),
        spaceAfter=8
    )
    
    meta_style = ParagraphStyle(
        "MetaStyle",
        parent=styles["BodyText"],
        fontName="Helvetica-Oblique",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#7f8c8d"),
        spaceAfter=15
    )
    
    story = []
    
    # Add Title
    story.append(Paragraph(title, title_style))
    
    # Add Metadata section
    meta_text = "<b>Experiment Metadata:</b><br/>"
    for k, v in metadata.items():
        meta_text += f"&bull; <b>{k}:</b> {v}<br/>"
    story.append(Paragraph(meta_text, meta_style))
    story.append(Spacer(1, 10))
    
    # Process content items
    for item in content_list:
        t = item["type"]
        if t == "h1":
            story.append(Paragraph(item["text"], h1_style))
        elif t == "h2":
            story.append(Paragraph(item["text"], h2_style))
        elif t == "p":
            story.append(Paragraph(item["text"], body_style))
        elif t == "spacer":
            story.append(Spacer(1, item.get("height", 10)))
        elif t == "page_break":
            story.append(PageBreak())
        elif t == "image":
            img_path = str(item["path"])
            if os.path.exists(img_path):
                # Scale image keeping aspect ratio
                story.append(Spacer(1, 5))
                story.append(Image(img_path, width=450, height=280))
                story.append(Spacer(1, 5))
        elif t == "table":
            df = item["data"]
            if isinstance(df, pd.DataFrame):
                # Convert DataFrame to list of lists, stringify all elements
                header = [str(c) for c in df.columns]
                rows = [[str(val) for val in row] for row in df.values]
                table_data = [header] + rows
            else:
                table_data = [[str(x) for x in r] for r in df]
                
            # Create ReportLab table
            col_widths = item.get("col_widths", None)
            rl_table = Table(table_data, colWidths=col_widths, hAlign="LEFT")
            
            # Styling Table matching medical journal layouts
            ts = TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f8f9fa")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 6),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 7.5),
                ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#2c3e50")),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 4),
                ("TOPPADDING", (0, 1), (-1, -1), 4),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("LINEBELOW", (0, 0), (-1, 0), 1.5, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fdfdfd")]),
            ])
            rl_table.setStyle(ts)
            story.append(Spacer(1, 5))
            story.append(rl_table)
            story.append(Spacer(1, 8))
            
    doc.build(story, canvasmaker=NumberedCanvas)
    return output_path
