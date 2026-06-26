#!/usr/bin/env python3
"""
ocr_converter.py — Convierte PDF a Markdown usando la mejor herramienta.
Usa ocr_classifier.py para decidir automáticamente.

Herramientas:
  - pymupdf4llm: PDFs digitales
  - ocrmypdf: PDFs escaneados simples
  - unlimited-ocr: PDFs largos/complejos

Uso:
  python3 ocr_converter.py <pdf_path> [output_md_path]
"""

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from ocr_classifier import classify_pdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def convert_with_pymupdf4llm(pdf_path: str, output_path: str) -> bool:
    """Convierte PDF a MD usando PyMuPDF4LLM (ideal para PDFs digitales)."""
    try:
        from pymupdf4llm import to_markdown

        logger.info("Convirtiendo con pymupdf4llm...")
        md = to_markdown(pdf_path)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)

        logger.info(f"✅ pymupdf4llm: {len(md)} chars escritos")
        return True

    except Exception as e:
        logger.error(f"❌ pymupdf4llm falló: {e}")
        return False


def convert_with_ocrmypdf(pdf_path: str, output_path: str, lang: str = "spa+eng") -> bool:
    """Convierte PDF a MD usando OCRmyPDF + Tesseract (ideal para escaneados)."""
    try:
        import ocrmypdf

        logger.info("Convirtiendo con ocrmypdf (Tesseract)...")

        # OCRmyPDF genera un PDF con capa de texto
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_pdf = tmp.name

        ocrmypdf.ocr(
            pdf_path,
            tmp_pdf,
            language=lang,
            output_type="pdf",
            force_ocr=True,
            optimize=1,
        )

        # Extraer texto del PDF resultante
        from pymupdf4llm import to_markdown
        md = to_markdown(tmp_pdf)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)

        # Limpiar temporal
        os.unlink(tmp_pdf)

        logger.info(f"✅ ocrmypdf: {len(md)} chars escritos")
        return True

    except Exception as e:
        logger.error(f"❌ ocrmypdf falló: {e}")
        return False


def convert_with_unlimited_ocr(pdf_path: str, output_path: str) -> bool:
    """Convierte PDF a MD usando Unlimited-OCR de Baidu (ideal para largos)."""
    try:
        from paddleocr import PaddleOCR
        import fitz

        logger.info("Convirtiendo con Unlimited-OCR (PaddleOCR)...")

        # Inicializar PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang="es", show_log=False)

        # Procesar cada página
        doc = fitz.open(pdf_path)
        all_text = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=150)

            # Guardar imagen temporal
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                img_path = tmp.name
                pix.save(img_path)

            # OCR
            result = ocr.ocr(img_path, cls=True)
            page_text = []
            for line in result:
                if line:
                    for word_info in line:
                        if word_info and len(word_info) >= 2:
                            page_text.append(word_info[1][0])

            all_text.append(f"## Página {page_num + 1}\n\n" + " ".join(page_text))

            # Limpiar
            os.unlink(img_path)

        doc.close()

        md = "\n\n".join(all_text)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)

        logger.info(f"✅ unlimited-ocr: {len(md)} chars escritos")
        return True

    except ImportError:
        logger.error("❌ PaddleOCR no instalado. Ejecuta: pip install paddleocr")
        return False
    except Exception as e:
        logger.error(f"❌ unlimited-ocr falló: {e}")
        return False


def convert_pdf_to_md(pdf_path: str, output_path: str = None) -> str:
    """
    Convierte PDF a MD automáticamente eligiendo la mejor herramienta.
    Retorna la ruta del archivo MD generado.
    """
    pdf_path = str(Path(pdf_path).resolve())

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF no encontrado: {pdf_path}")

    if output_path is None:
        output_path = str(Path(pdf_path).with_suffix(".md"))

    # Clasificar
    classification = classify_pdf(pdf_path)
    tool = classification["tool"]
    reason = classification["reason"]
    confidence = classification["confidence"]

    logger.info(f"🤖 Clasificador: {tool} (confianza: {confidence})")
    logger.info(f"   Razón: {reason}")

    # Convertir
    success = False
    if tool == "pymupdf4llm":
        success = convert_with_pymupdf4llm(pdf_path, output_path)
        # Fallback si falla
        if not success:
            logger.info("🔄 Fallback: probando ocrmypdf...")
            success = convert_with_ocrmypdf(pdf_path, output_path)

    elif tool == "ocrmypdf":
        success = convert_with_ocrmypdf(pdf_path, output_path)
        if not success:
            logger.info("🔄 Fallback: probando pymupdf4llm...")
            success = convert_with_pymupdf4llm(pdf_path, output_path)

    elif tool == "unlimited-ocr":
        success = convert_with_unlimited_ocr(pdf_path, output_path)
        if not success:
            logger.info("🔄 Fallback: probando pymupdf4llm...")
            success = convert_with_pymupdf4llm(pdf_path, output_path)

    if success:
        size_kb = os.path.getsize(output_path) / 1024
        logger.info(f"✅ Conversión exitosa: {output_path} ({size_kb:.1f} KB)")
        return output_path
    else:
        raise RuntimeError(f"No se pudo convertir {pdf_path} con ninguna herramienta")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 ocr_converter.py <pdf_path> [output_md_path]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    result = convert_pdf_to_md(pdf_path, output_path)
    print(result)
