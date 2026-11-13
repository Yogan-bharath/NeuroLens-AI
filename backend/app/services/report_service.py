import io
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

from app.schemas.analysis import AnalysisResponse


class ReportService:
    @staticmethod
    def generate_pdf_report(analysis: AnalysisResponse) -> bytes:
        """Generate clinical-grade research report PDF in memory."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=40,
            leftMargin=40,
            topMargin=40,
            bottomMargin=40
        )
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            "ReportTitle",
            parent=styles["Heading1"],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#0f172a"),
            fontName="Helvetica-Bold",
            spaceAfter=4,
        )
        subtitle_style = ParagraphStyle(
            "ReportSubtitle",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#64748b"),
            spaceAfter=15,
        )
        disclaimer_style = ParagraphStyle(
            "Disclaimer",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#dc2626"),
            fontName="Helvetica-Bold",
        )
        heading2_style = ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#0f172a"),
            spaceBefore=12,
            spaceAfter=6,
        )
        cell_style = ParagraphStyle(
            "CellText",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#334155"),
        )

        elements = []

        # 1. Header
        elements.append(Paragraph("NeuroLens AI — Brain MRI Research Report", title_style))
        elements.append(Paragraph("Automated Deep Learning Classification, Segmentation & Explainability Record", subtitle_style))

        # 2. Disclaimer Box
        disclaimer_text = (
            "<b>IMPORTANT REGULATORY NOTICE:</b> This document contains algorithmic outputs generated "
            "by artificial intelligence models for investigative research use only. This report is NOT a medical diagnosis, "
            "does NOT constitute clinical advice, and must NOT replace qualified neuroradiological evaluation."
        )
        disclaimer_table = Table([[Paragraph(disclaimer_text, disclaimer_style)]], colWidths=[532])
        disclaimer_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef2f2")),
            ("BORDER", (0, 0), (-1, -1), 1, colors.HexColor("#f87171")),
            ("PADDING", (0, 0), (-1, -1), 8),
        ]))
        elements.append(disclaimer_table)
        elements.append(Spacer(1, 15))

        # 3. Session Metadata Table
        meta_data = [
            [Paragraph("<b>Analysis Session ID:</b>", cell_style), Paragraph(analysis.id, cell_style),
             Paragraph("<b>Analysis Mode:</b>", cell_style), Paragraph(analysis.analysis_mode.capitalize(), cell_style)],
            [Paragraph("<b>Original File:</b>", cell_style), Paragraph(analysis.fileName, cell_style),
             Paragraph("<b>Execution Date:</b>", cell_style), Paragraph(analysis.createdAt[:19].replace("T", " ") + " UTC", cell_style)],
            [Paragraph("<b>Primary Model:</b>", cell_style), Paragraph(analysis.metadata.modelName if analysis.metadata else "N/A", cell_style),
             Paragraph("<b>Processing Duration:</b>", cell_style), Paragraph(f"{analysis.metadata.processingTimeMs} ms" if analysis.metadata else "N/A", cell_style)],
        ]
        meta_table = Table(meta_data, colWidths=[120, 146, 120, 146])
        meta_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("PADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(meta_table)
        elements.append(Spacer(1, 10))

        # 4. Classification Results
        if analysis.classification:
            elements.append(Paragraph("Classification Predictions", heading2_style))
            cls_rows = [[Paragraph("<b>Diagnostic Class</b>", cell_style), Paragraph("<b>Probability Score</b>", cell_style), Paragraph("<b>Prediction Status</b>", cell_style)]]
            for item in analysis.classification:
                status_txt = "<b>TOP PREDICTION</b>" if item.isTopPrediction else "Alternative"
                cls_rows.append([
                    Paragraph(item.label, cell_style),
                    Paragraph(f"{item.score * 100:.2f}%", cell_style),
                    Paragraph(status_txt, cell_style)
                ])
            cls_table = Table(cls_rows, colWidths=[200, 160, 172])
            cls_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]))
            elements.append(cls_table)
            elements.append(Spacer(1, 10))

        # 5. Segmentation Results
        if analysis.segmentation:
            elements.append(Paragraph("Lesion Segmentation Analysis (U-Net)", heading2_style))
            seg_rows = [[Paragraph("<b>Region Label</b>", cell_style), Paragraph("<b>Pixel Count</b>", cell_style), Paragraph("<b>Area Fraction</b>", cell_style)]]
            for reg in analysis.segmentation.classes:
                seg_rows.append([
                    Paragraph(reg.label, cell_style),
                    Paragraph(f"{reg.pixelCount:,} px", cell_style),
                    Paragraph(f"{reg.percentage:.2f}%", cell_style)
                ])
            seg_table = Table(seg_rows, colWidths=[200, 160, 172])
            seg_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]))
            elements.append(seg_table)
            elements.append(Spacer(1, 10))

        # 6. Explainability Methods
        if analysis.explanations:
            elements.append(Paragraph("Explainable AI (XAI) Attribution Methods", heading2_style))
            xai_rows = [[Paragraph("<b>Method</b>", cell_style), Paragraph("<b>Target Layer</b>", cell_style), Paragraph("<b>Attribution Focus</b>", cell_style)]]
            for exp in analysis.explanations:
                xai_rows.append([
                    Paragraph(exp.method.upper(), cell_style),
                    Paragraph(exp.layerName, cell_style),
                    Paragraph(exp.targetClass, cell_style)
                ])
            xai_table = Table(xai_rows, colWidths=[150, 150, 232])
            xai_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]))
            elements.append(xai_table)
            elements.append(Spacer(1, 15))

        # 7. Signature Footer
        elements.append(Paragraph("Generated by NeuroLens AI Research Platform — Confidential Medical Informatics Document", subtitle_style))

        doc.build(elements)
        buffer.seek(0)
        return buffer.getvalue()


report_service = ReportService()
