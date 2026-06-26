#!/usr/bin/env python3
"""
ocr_converter.py — Convierte PDF a Markdown usando la mejor herramienta.
Usa ocr_classifier.py para decidir automáticamente.

Herramientas:
  - pymupdf4llm: PDFs digitales
  - ocrmypdf: PDFs escaneados simples
  - unlimited-ocr: PDFs largos/complejos (PaddleOCR directo)
  - paddleocr-plugin: PDFs escaneados en español (via ocrmypdf-paddleocr)  [MEJORA 2]
  - docai-pipeline: Pipeline completo PDF → Layout Analysis → OCR → Markdown [MEJORA 5]

Uso:
  python3 ocr_converter.py <pdf_path> [output_md_path]
"""

import json
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


# =============================================================================
# MEJORA 2: PaddleOCR plugin para OCRmyPDF
# =============================================================================

def convert_with_paddleocr(pdf_path: str, output_path: str, lang: str = "spa+eng") -> bool:
    """
    Convierte PDF a MD usando ocrmypdf con plugin PaddleOCR.
    Ideal para PDFs escaneados en español.
    Requiere: pip install ocrmypdf-paddleocr

    Args:
        pdf_path: Ruta al PDF
        output_path: Ruta de salida del archivo MD
        lang: Idioma (default: spa+eng)

    Returns:
        True si tuvo éxito, False en caso contrario
    """
    try:
        import ocrmypdf
        import pymupdf4llm

        logger.info("Convirtiendo con ocrmypdf + PaddleOCR plugin...")

        # Verificar que el plugin paddleocr está disponible
        try:
            from ocrmypdf_paddleocr import PaddleOcrPlugin
            plugin = PaddleOcrPlugin()
            logger.info("✅ Plugin PaddleOCR cargado correctamente")
        except ImportError:
            logger.warning("⚠️  Plugin PaddleOCR no disponible, usando Tesseract como fallback")
            return convert_with_ocrmypdf(pdf_path, output_path, lang)

        # OCRmyPDF con PaddleOCR
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_pdf = tmp.name

        ocrmypdf.ocr(
            pdf_path,
            tmp_pdf,
            language=lang,
            output_type="pdf",
            force_ocr=True,
            optimize=1,
            plugin=plugin,
            progress_bar=False,
        )

        # Extraer texto del PDF resultante
        md = pymupdf4llm.to_markdown(tmp_pdf)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md)

        # Limpiar temporal
        os.unlink(tmp_pdf)

        logger.info(f"✅ ocrmypdf+PaddleOCR: {len(md)} chars escritos")
        return True

    except ImportError as e:
        logger.warning(f"⚠️  Dependencia faltante para PaddleOCR ({e}), usando fallback Tesseract...")
        return convert_with_ocrmypdf(pdf_path, output_path, lang)
    except Exception as e:
        logger.error(f"❌ PaddleOCR falló: {e}")
        logger.info("🔄 Fallback: probando ocrmypdf (Tesseract)...")
        return convert_with_ocrmypdf(pdf_path, output_path, lang)


# =============================================================================
# MEJORA 5: docAI toolkit — Pipeline completo PDF → Layout Analysis → OCR → Markdown
# =============================================================================

def process_document_pipeline(pdf_path: str, output_path: str = None) -> dict:
    """
    Pipeline completo de procesamiento de documentos.
    PDF → Layout Analysis → OCR → Markdown → Chunks listos para indexar.

    Esta función utiliza componentes ligeros disponibles localmente:
      1. PyMuPDF para layout analysis (detección de párrafos, tablas, imágenes)
      2. PaddleOCR o Tesseract para OCR de páginas escaneadas
      3. pymupdf4llm para conversión a Markdown preservando estructura
      4. Chunking semántico para preparar chunks

    Args:
        pdf_path: Ruta al PDF
        output_path: Ruta de salida (opcional, default: pdf_path + "_pipeline.md")

    Returns:
        dict con:
          - "markdown_path": ruta al archivo MD generado
          - "chunks": lista de chunks listos para indexar
          - "pages": número de páginas procesadas
          - "layout": info del layout detectado

    Nota: Esta es una opción premium que no reemplaza el flujo estándar.
    """
    import fitz  # PyMuPDF

    if output_path is None:
        output_path = str(Path(pdf_path).with_suffix("") + "_pipeline.md")

    logger.info(f"📄 Iniciando docAI pipeline: {pdf_path}")
    logger.info(f"   Pipeline: Layout Analysis → OCR → Markdown → Chunks")

    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    # 1. Layout Analysis
    layout_info = {
        "total_pages": total_pages,
        "pages_with_text": 0,
        "pages_with_images": 0,
        "pages_with_tables": 0,
        "paragraphs": 0,
        "images": 0,
    }

    page_analyses = []
    for page_num in range(total_pages):
        page = doc[page_num]

        # Analizar bloques de texto (párrafos)
        blocks = page.get_text("blocks")
        text_blocks = [b for b in blocks if b[6] == 0]  # block type 0 = text
        image_blocks = [b for b in blocks if b[6] == 1]  # block type 1 = image

        page_has_text = len(text_blocks) > 0
        page_has_images = len(image_blocks) > 0

        # Detección simple de tablas (bloques con estructura de columnas)
        if page_has_text:
            # Buscar bloques alineados verticalmente (posible tabla)
            x_positions = set()
            for b in text_blocks:
                x_positions.add(round(b[0], -1))  # redondear x0 a decena
            has_table_like = len(x_positions) > 3 and len(text_blocks) > 5
        else:
            has_table_like = False

        if page_has_text:
            layout_info["pages_with_text"] += 1
        if page_has_images:
            layout_info["pages_with_images"] += 1
        if has_table_like:
            layout_info["pages_with_tables"] += 1

        layout_info["paragraphs"] += len(text_blocks)
        layout_info["images"] += len(image_blocks)

        page_analyses.append({
            "page": page_num + 1,
            "text_blocks": len(text_blocks),
            "image_blocks": len(image_blocks),
            "has_table": has_table_like,
        })

    doc.close()
    logger.info(f"   Layout: {layout_info['pages_with_text']} páginas con texto, "
                f"{layout_info['pages_with_images']} con imágenes, "
                f"{layout_info['paragraphs']} párrafos")

    # 2. Conversión a Markdown (usando pipeline existente)
    logger.info(f"   Ejecutando OCR/extracción...")
    try:
        from pymupdf4llm import to_markdown
        markdown_content = to_markdown(pdf_path)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        logger.info(f"   ✅ Markdown generado: {len(markdown_content)} chars")
    except Exception as e:
        logger.warning(f"   ⚠️  pymupdf4llm falló ({e}), intentando OCR...")
        # Fallback: ocrmypdf + extracción
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_pdf = tmp.name
        try:
            import ocrmypdf
            ocrmypdf.ocr(pdf_path, tmp_pdf, language="spa+eng", force_ocr=True, optimize=1)
            from pymupdf4llm import to_markdown
            markdown_content = to_markdown(tmp_pdf)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown_content)
        finally:
            if os.path.exists(tmp_pdf):
                os.unlink(tmp_pdf)

    # 3. Chunking semántico (reutilizando chunk_text de turbovec_rag)
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from turbovec_rag import chunk_text
        chunks = chunk_text(markdown_content, chunk_size=500, overlap=100)
        logger.info(f"   ✅ {len(chunks)} chunks generados vía chunking semántico")
    except Exception as e:
        logger.warning(f"   ⚠️  chunking falló ({e})")
        chunks = [markdown_content]

    result = {
        "markdown_path": output_path,
        "chunks": chunks,
        "pages": total_pages,
        "layout": layout_info,
        "page_analyses": page_analyses,
    }

    logger.info(f"✅ docAI pipeline completado: {output_path}")
    return result


# =============================================================================
# Funciones de conversión existentes (sin cambios)
# =============================================================================

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

    El clasificador ahora puede recomendar "paddleocr" para PDFs escaneados
    en español (MEJORA 2).
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
        if not success:
            logger.info("🔄 Fallback: probando ocrmypdf...")
            success = convert_with_ocrmypdf(pdf_path, output_path)

    elif tool == "ocrmypdf":
        # --- MEJORA 2: Preferir PaddleOCR para PDFs escaneados en español ---
        if classification.get("prefer_paddleocr", False):
            logger.info("🔤 PDF escaneado en español — probando PaddleOCR primero...")
            success = convert_with_paddleocr(pdf_path, output_path)
            if not success:
                logger.info("🔄 Fallback: probando ocrmypdf (Tesseract)...")
                success = convert_with_ocrmypdf(pdf_path, output_path)
        else:
            success = convert_with_ocrmypdf(pdf_path, output_path)
        if not success:
            logger.info("🔄 Fallback: probando pymupdf4llm...")
            success = convert_with_pymupdf4llm(pdf_path, output_path)

    elif tool == "unlimited-ocr":
        success = convert_with_unlimited_ocr(pdf_path, output_path)
        if not success:
            logger.info("🔄 Fallback: probando pymupdf4llm...")
            success = convert_with_pymupdf4llm(pdf_path, output_path)

    elif tool == "paddleocr":
        # --- MEJORA 2: Nueva herramienta "paddleocr" ---
        success = convert_with_paddleocr(pdf_path, output_path)
        if not success:
            logger.info("🔄 Fallback: probando ocrmypdf (Tesseract)...")
            success = convert_with_ocrmypdf(pdf_path, output_path)

    if success:
        size_kb = os.path.getsize(output_path) / 1024
        logger.info(f"✅ Conversión exitosa: {output_path} ({size_kb:.1f} KB)")
        return output_path
    else:
        raise RuntimeError(f"No se pudo convertir {pdf_path} con ninguna herramienta")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 ocr_converter.py <pdf_path> [output_md_path]")
        print("       python3 ocr_converter.py --pipeline <pdf_path> [output_md_path]")
        sys.exit(1)

    if sys.argv[1] == "--pipeline":
        # --- MEJORA 5: docAI pipeline ---
        pdf_path = sys.argv[2]
        output_path = sys.argv[3] if len(sys.argv) > 3 else None
        result = process_document_pipeline(pdf_path, output_path)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        pdf_path = sys.argv[1]
        output_path = sys.argv[2] if len(sys.argv) > 2 else None
        result = convert_pdf_to_md(pdf_path, output_path)
        print(result)
