#!/usr/bin/env python3
"""
ocr_classifier.py — Agente clasificador de OCR para PDFs.
Analiza el contenido del PDF y decide la mejor herramienta de conversión.

Herramientas disponibles:
  - pymupdf4llm: PDFs digitales (texto seleccionable), rápido, CPU-friendly
  - ocrmypdf (Tesseract): PDFs escaneados simples, multilingüe
  - unlimited-ocr (Baidu): PDFs largos/complejos, requiere más recursos

Uso:
  python3 ocr_classifier.py <pdf_path>
"""

import json
import logging
import os
import sys
from pathlib import Path

import fitz  # PyMuPDF

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Umbrales
MIN_TEXT_RATIO_DIGITAL = 0.3  # 30% de páginas con texto = digital
MIN_TEXT_PER_PAGE = 50  # caracteres mínimos por página para considerar digital
MAX_PAGES_FOR_OCRMYPDF = 100  # OCRmyPDF funciona bien hasta ~100 páginas
MIN_PAGES_FOR_UNLIMITED_OCR = 50  # Más de 50 páginas → Unlimited-OCR


class PDFCAnalyzer:
    """Analiza el contenido de un PDF para clasificarlo."""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.stats = self._analyze()

    def _analyze(self) -> dict:
        """Analiza página por página el PDF."""
        total_pages = len(self.doc)
        pages_with_text = 0
        pages_without_text = 0
        pages_with_images = 0
        total_text_chars = 0
        text_page_ratios = []

        for page_num in range(total_pages):
            page = self.doc[page_num]

            # Extraer texto
            text = page.get_text().strip()
            text_len = len(text)
            total_text_chars += text_len

            # Detectar imágenes
            images = page.get_images()
            has_images = len(images) > 0

            if text_len > MIN_TEXT_PER_PAGE:
                pages_with_text += 1
            else:
                pages_without_text += 1

            if has_images:
                pages_with_images += 1

            # Ratio texto vs área de página
            page_area = page.rect.width * page.rect.height
            if page_area > 0:
                text_ratio = text_len / (page_area / 1000)  # chars por 1000px²
            else:
                text_ratio = 0
            text_page_ratios.append(text_ratio)

        # Promedio
        avg_text_per_page = total_text_chars / total_pages if total_pages > 0 else 0
        text_page_ratio = pages_with_text / total_pages if total_pages > 0 else 0

        return {
            "total_pages": total_pages,
            "pages_with_text": pages_with_text,
            "pages_without_text": pages_without_text,
            "pages_with_images": pages_with_images,
            "total_text_chars": total_text_chars,
            "avg_text_per_page": avg_text_per_page,
            "text_page_ratio": text_page_ratio,
            "has_mixed_content": pages_with_images > 0 and pages_with_text > 0,
        }

    def classify(self) -> dict:
        """Decide la herramienta óptima para este PDF."""
        stats = self.stats
        total_pages = stats["total_pages"]
        text_ratio = stats["text_page_ratio"]
        pages_without_text = stats["pages_without_text"]
        has_images = stats["pages_with_images"] > 0

        # Caso 1: PDF digital (texto seleccionable)
        if text_ratio >= MIN_TEXT_RATIO_DIGITAL and pages_without_text < total_pages * 0.2:
            return {
                "tool": "pymupdf4llm",
                "reason": f"PDF digital: {stats['pages_with_text']}/{total_pages} páginas con texto",
                "confidence": "high",
                "stats": stats,
            }

        # Caso 2: PDF escaneado simple (pocas páginas, mayormente imágenes)
        if total_pages <= MAX_PAGES_FOR_OCRMYPDF and pages_without_text > total_pages * 0.5:
            return {
                "tool": "ocrmypdf",
                "reason": f"Escaneado simple: {pages_without_text}/{total_pages} páginas sin texto, {total_pages} páginas total",
                "confidence": "high",
                "stats": stats,
            }

        # Caso 3: PDF largo/complejo (muchas páginas + escaneado)
        if total_pages > MIN_PAGES_FOR_UNLIMITED_OCR:
            return {
                "tool": "unlimited-ocr",
                "reason": f"PDF largo: {total_pages} páginas, requiere procesamiento en una pasada",
                "confidence": "medium",
                "stats": stats,
            }

        # Caso 4: Contenido mixto (texto + imágenes)
        if stats["has_mixed_content"]:
            return {
                "tool": "pymupdf4llm",
                "reason": f"Contenido mixto: {stats['pages_with_images']} páginas con imágenes + texto",
                "confidence": "medium",
                "stats": stats,
            }

        # Caso por defecto
        return {
            "tool": "pymupdf4llm",
            "reason": f"Caso general: ratio texto={text_ratio:.2f}, {total_pages} páginas",
            "confidence": "low",
            "stats": stats,
        }

    def close(self):
        self.doc.close()


def classify_pdf(pdf_path: str) -> dict:
    """Función helper para clasificar un PDF."""
    analyzer = PDFCAnalyzer(pdf_path)
    result = analyzer.classify()
    analyzer.close()
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 ocr_classifier.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"❌ Archivo no encontrado: {pdf_path}")
        sys.exit(1)

    result = classify_pdf(pdf_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))
