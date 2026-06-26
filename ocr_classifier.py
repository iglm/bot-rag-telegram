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

# --- MEJORA 2: Palabras clave para detectar PDFs en español ---
SPANISH_KEYWORDS = [
    "el", "la", "los", "las", "de", "del", "que", "en", "por", "para",
    "con", "sin", "es", "son", "se", "su", "sus", "entre", "sobre",
    "este", "esta", "estos", "estas", "como", "más", "pero", "tiene",
    "una", "uno", "fecha", "total", "código", "nombre", "dirección",
    "documento", "factura", "informe", "contrato", "certificado",
    "español", "colombia", "señor", "señora", "gracias", "favor",
]

# --- MEJORA 1+5: Verificar disponibilidad de parsers alternativos ---
HAS_MINERU_CHECK = False
try:
    import importlib
    if importlib.util.find_spec("mineru") is not None:
        HAS_MINERU_CHECK = True
except Exception:
    pass

HAS_PDF_OXIDE_CHECK = False
try:
    from pdf_oxide import PdfDocument
    HAS_PDF_OXIDE_CHECK = True
except ImportError:
    pass


class PDFCAnalyzer:
    """Analiza el contenido de un PDF para clasificarlo."""

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        self.stats = self._analyze()

    def _detect_spanish(self, sample_text: str) -> bool:
        """Detecta si el texto del PDF es principalmente español."""
        if not sample_text.strip():
            return False
        text_lower = sample_text.lower()
        spanish_count = sum(1 for kw in SPANISH_KEYWORDS if kw in text_lower)
        # Si encontramos al menos 3 keywords españolas, asumimos español
        return spanish_count >= 3

    def _detect_tables_or_formulas(self, sample_text: str, stats: dict) -> bool:
        """Detecta si el PDF contiene tablas, fórmulas o estructura de columnas."""
        if not sample_text:
            return False
        text_lower = sample_text.lower()
        # Indicadores de tablas
        table_indicators = ["|", "\t", "columna", "fila", "tabla", "table",
                           "datos", "valores", "medición", "medicion", "promedio"]
        # Indicadores de fórmulas matemáticas
        formula_indicators = ["=", "+", "-", "*", "/", "∑", "∫", "π", "Δ", "θ",
                             "fórmula", "formula", "ecuación", "ecuacion"]
        has_tables = any(ind in text_lower for ind in table_indicators)
        has_formulas = any(ind in text_lower for ind in formula_indicators)
        # Múltiples columnas detectadas por análisis de layout
        has_columns = stats.get("avg_text_per_page", 0) > 2000 and stats.get("total_pages", 0) > 5
        return has_tables or has_formulas or has_columns

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

        # Obtener muestra de texto para detección de idioma
        sample_text = ""
        try:
            for page_num in range(min(3, total_pages)):
                sample_text += self.doc[page_num].get_text().strip() + " "
        except Exception:
            pass

        is_spanish = self._detect_spanish(sample_text)

        # Caso 1: PDF digital (texto seleccionable)
        if text_ratio >= MIN_TEXT_RATIO_DIGITAL and pages_without_text < total_pages * 0.2:
            # --- MEJORA 1+5: Clasificar sub-tipos de PDFs digitales ---
            # PDFs con tablas, fórmulas o muchas columnas → MinerU
            has_tables_or_formulas = self._detect_tables_or_formulas(sample_text, stats)
            # PDFs bien formados sin estructura compleja → PDF Oxide
            is_simple_digital = (
                total_pages <= 50
                and not has_tables_or_formulas
                and stats["total_text_chars"] > 1000
            )

            if has_tables_or_formulas and HAS_MINERU_CHECK:
                return {
                    "tool": "pymupdf4llm",
                    "reason": f"PDF digital complejo (tablas/fórmulas): {stats['pages_with_text']}/{total_pages} páginas con texto",
                    "confidence": "high",
                    "stats": stats,
                    "prefer_paddleocr": False,
                    "prefer_mineru": True,
                    "prefer_pdf_oxide": False,
                }
            elif is_simple_digital and HAS_PDF_OXIDE_CHECK:
                return {
                    "tool": "pymupdf4llm",
                    "reason": f"PDF digital simple: {stats['pages_with_text']}/{total_pages} páginas con texto",
                    "confidence": "high",
                    "stats": stats,
                    "prefer_paddleocr": False,
                    "prefer_mineru": False,
                    "prefer_pdf_oxide": True,
                }
            else:
                return {
                    "tool": "pymupdf4llm",
                    "reason": f"PDF digital: {stats['pages_with_text']}/{total_pages} páginas con texto",
                    "confidence": "high",
                    "stats": stats,
                    "prefer_paddleocr": False,
                    "prefer_mineru": False,
                    "prefer_pdf_oxide": False,
                }

        # --- MEJORA 2: Preferir PaddleOCR para PDFs escaneados en español ---
        if is_spanish and pages_without_text > total_pages * 0.3:
            return {
                "tool": "paddleocr",
                "reason": f"PDF escaneado en español: {pages_without_text}/{total_pages} páginas sin texto",
                "confidence": "high",
                "stats": stats,
                "prefer_paddleocr": True,
                "prefer_mineru": False,
                "prefer_pdf_oxide": False,
            }

        # Caso 2: PDF escaneado simple (pocas páginas, mayormente imágenes)
        if total_pages <= MAX_PAGES_FOR_OCRMYPDF and pages_without_text > total_pages * 0.5:
            return {
                "tool": "ocrmypdf",
                "reason": f"Escaneado simple: {pages_without_text}/{total_pages} páginas sin texto, {total_pages} páginas total",
                "confidence": "high",
                "stats": stats,
                "prefer_paddleocr": is_spanish,
                "prefer_mineru": False,
                "prefer_pdf_oxide": False,
            }

        # Caso 3: PDF largo/complejo (muchas páginas + escaneado)
        if total_pages > MIN_PAGES_FOR_UNLIMITED_OCR:
            return {
                "tool": "unlimited-ocr",
                "reason": f"PDF largo: {total_pages} páginas, requiere procesamiento en una pasada",
                "confidence": "medium",
                "stats": stats,
                "prefer_paddleocr": is_spanish,
                "prefer_mineru": False,
                "prefer_pdf_oxide": False,
            }

        # Caso 4: Contenido mixto (texto + imágenes)
        if stats["has_mixed_content"]:
            return {
                "tool": "pymupdf4llm",
                "reason": f"Contenido mixto: {stats['pages_with_images']} páginas con imágenes + texto",
                "confidence": "medium",
                "stats": stats,
                "prefer_paddleocr": False,
                "prefer_mineru": False,
                "prefer_pdf_oxide": False,
            }

        # Caso por defecto
        return {
            "tool": "pymupdf4llm",
            "reason": f"Caso general: ratio texto={text_ratio:.2f}, {total_pages} páginas",
            "confidence": "low",
            "stats": stats,
            "prefer_paddleocr": False,
            "prefer_mineru": False,
            "prefer_pdf_oxide": False,
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
