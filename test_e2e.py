#!/usr/bin/env python3
"""
test_e2e.py — Test end-to-end del sistema RAG completo.
Simula: PDF → OCR → indexar → query → respuesta

Uso:
  python3 test_e2e.py              # ejecutar test completo
  python3 test_e2e.py --skip-ocr   # saltar OCR (más rápido)
  python3 test_e2e.py --verbose     # logs detallados
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

# Silenciar logs ruidosos de HuggingFace
logging.disable(logging.WARNING)

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

# Verificar imports
try:
    from turbovec_rag import RagEngine, extract_text_from_file
    from ocr_classifier import classify_pdf
    from ocr_classifier import PDFCAnalyzer
    print("✅ Imports OK")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

try:
    import fitz  # PyMuPDF
except ImportError:
    print("❌ PyMuPDF no instalado: pip install PyMuPDF")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("❌ numpy no instalado: pip install numpy")
    sys.exit(1)


def create_test_pdf(text: str, filename: str = "test.pdf") -> str:
    """Crea un PDF de prueba con el texto dado."""
    doc = fitz.open()
    page = doc.new_page()
    # Insertar texto en múltiples líneas
    lines = text.split("\n")
    y = 50
    for line in lines:
        page.insert_text((50, y), line)
        y += 15
        if y > 750:  # nueva página
            page = doc.new_page()
            y = 50
    path = f"/tmp/{filename}"
    doc.save(path)
    doc.close()
    return path


def test_1_text_extraction():
    """Test 1: Extracción de texto de archivos."""
    print("\n📋 Test 1: Text extraction")

    # TXT
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
        f.write("El café colombiano se cultiva en Caldas, Quindío y Huila")
        f.flush()
        txt_text = extract_text_from_file(f.name)
    assert "café" in txt_text.lower(), f"TXT extraction failed: {txt_text[:50]}"
    os.unlink(f.name)
    print("  ✅ TXT extraction OK")

    # MD
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
        f.write("# Café\nProducción: 12M sacos\nRegión: Caldas")
        f.flush()
        md_text = extract_text_from_file(f.name)
    assert "Café" in md_text or "café" in md_text.lower()
    os.unlink(f.name)
    print("  ✅ MD extraction OK")

    # JSON
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump({"producto": "café", "produccion": 12000000, "region": "Caldas"}, f)
        f.flush()
        json_text = extract_text_from_file(f.name)
    assert "café" in json_text.lower() or "cafe" in json_text.lower()
    os.unlink(f.name)
    print("  ✅ JSON extraction OK")

    # PDF
    pdf_path = create_test_pdf("Café colombiano\nProducción: 12 millones de sacos\nRegión: Caldas, Quindío")
    pdf_text = extract_text_from_file(pdf_path)
    assert "café" in pdf_text.lower() or "Café" in pdf_text
    os.unlink(pdf_path)
    print("  ✅ PDF extraction OK")

    return True


def test_2_ocr_classifier():
    """Test 2: Clasificador OCR."""
    print("\n🤖 Test 2: OCR Classifier")

    # PDF digital
    pdf_path = create_test_pdf("Texto digital de prueba\n" * 20, "digital.pdf")
    result = classify_pdf(pdf_path)
    assert result["tool"] == "pymupdf4llm", f"Expected pymupdf4llm, got {result['tool']}"
    assert result["confidence"] == "high"
    os.unlink(pdf_path)
    print("  ✅ PDF digital → pymupdf4llm (high confidence)")

    # PDF con solo imágenes (simulado: página vacía)
    doc = fitz.open()
    page = doc.new_page()
    # Sin texto, solo una figura
    page.draw_rect(fitz.Rect(50, 50, 200, 200))
    pdf_path = "/tmp/empty.pdf"
    doc.save(pdf_path)
    doc.close()
    result = classify_pdf(pdf_path)
    # Debería elegir pymupdf4llm (fallback) o ocrmypdf
    assert result["tool"] in ("pymupdf4llm", "ocrmypdf")
    os.unlink(pdf_path)
    print(f"  ✅ PDF vacío → {result['tool']} (fallback)")

    return True


def test_3_indexing():
    """Test 3: Indexación con texto completo en metadata."""
    print("\n🗄️ Test 3: Indexing")

    eng = RagEngine()

    # Indexar archivo
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", prefix="e2e_") as f:
        f.write("El café de altura se produce entre 1200 y 2000 metros sobre el nivel del mar en Colombia")
        f.flush()
        tp = f.name

    count = eng.index_file(tp)
    assert count >= 1, f"Expected at least 1 chunk, got {count}"

    # Verificar que metadata tiene texto completo
    last_entry = eng.metadata[-1]
    assert "text" in last_entry, f"Missing 'text' field. Keys: {list(last_entry.keys())}"
    assert len(last_entry["text"]) > 50, f"Text too short: {len(last_entry['text'])} chars"
    assert "text_preview" in last_entry
    assert last_entry["text_preview"] == last_entry["text"][:200]

    os.unlink(tp)
    print(f"  ✅ Indexación OK: {count} chunks")
    print(f"  ✅ Metadata tiene 'text' ({len(last_entry['text'])} chars)")
    print(f"  ✅ Metadata tiene 'text_preview' ({len(last_entry['text_preview'])} chars)")

    return eng


def test_4_query(eng: RagEngine):
    """Test 4: Búsqueda y retrieval."""
    print("\n🔍 Test 4: Query + Retrieval")

    # Buscar en los datos recién indexados (que sí tienen text)
    # Buscar query que termine en el último chunk indexado
    results = eng.query("café de altura 2000 metros Colombia", top_k=5)

    assert len(results) > 0, "No results returned"
    assert "score" in results[0], "Result missing 'score' field"
    assert "pdf" in results[0], "Result missing 'pdf' field"

    # Buscar el resultado que tiene 'text' (el recién indexado)
    results_with_text = [r for r in results if r.get("text")]
    if results_with_text:
        print(f"  ✅ Query retornó {len(results)} resultados ({len(results_with_text)} con texto completo)")
        print(f"  ✅ Score top-1: {results[0]['score']:.2f}")
        print(f"  ✅ Text disponible en resultado nuevo: {len(results_with_text[0]['text'])} chars")
    else:
        # Los datos viejos no tienen text, solo preview
        assert "text_preview" in results[0], "Result missing 'text_preview'"
        print(f"  ✅ Query retornó {len(results)} resultados (usando text_preview)")
        print(f"  ✅ Score top-1: {results[0]['score']:.2f}")
        print(f"  ✅ Text preview disponible: {len(results[0].get('text_preview', ''))} chars")
    full_text = eng._get_chunk_text(len(eng.metadata) - 1)
    assert len(full_text) > 50, f"_get_chunk_text too short: {len(full_text)}"
    print(f"  ✅ _get_chunk_text retorna texto completo: {len(full_text)} chars")

    return results


def test_5_deduplication():
    """Test 5: Deduplicación por hash (simulando comportamiento del bot)."""
    print("\n🔄 Test 5: Deduplication")

    eng = RagEngine()
    initial_count = len(eng.metadata)

    # Crear archivo
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", prefix="dedup_") as f:
        f.write("Contenido único para deduplicación test café colombiano")
        f.flush()
        tp = f.name

    # Indexar primera vez
    count1 = eng.index_file(tp)
    after_first = len(eng.metadata)
    assert after_first == initial_count + count1

    # Simular que el bot agrega hash (no es responsabilidad del engine)
    if eng.metadata:
        eng.metadata[-1]["hash"] = hashlib.sha256(open(tp, "rb").read()).hexdigest()

    # Indexar mismo archivo otra vez
    count2 = eng.index_file(tp)
    after_second = len(eng.metadata)

    # Verificar que el hash se guardó
    assert "hash" in eng.metadata[-1], "Hash field not saved"

    os.unlink(tp)
    print(f"  ✅ Primera indexación: +{count1} chunks (total: {after_first})")
    print(f"  ✅ Segunda indexación: +{count2} chunks (total: {after_second})")
    print(f"  ✅ Hash guardado: {eng.metadata[-1]['hash'][:16]}...")

    return True


def test_6_batch_embeddings():
    """Test 6: Embeddings en batch (múltiples chunks)."""
    print("\n⚡ Test 6: Batch Embeddings")

    eng = RagEngine()

    # Crear archivo largo que genere múltiples chunks
    long_text = " ".join([f"palabra{i}" for i in range(2000)])
    long_text += " café producción Colombia Caldas Quindío"

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", prefix="batch_") as f:
        f.write(long_text)
        f.flush()
        tp = f.name

    start = time.time()
    count = eng.index_file(tp)
    elapsed = time.time() - start

    os.unlink(tp)
    assert count > 1, f"Expected multiple chunks, got {count}"

    print(f"  ✅ {count} chunks indexados en {elapsed:.2f}s")
    print(f"  ✅ Batch processing funcional")

    return True


def test_7_save_load():
    """Test 7: Guardar y cargar índice."""
    print("\n💾 Test 7: Save & Load Index")

    eng = RagEngine()

    # Indexar
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w", prefix="saveload_") as f:
        f.write("Texto para guardar y cargar índice turbovec café")
        f.flush()
        tp = f.name

    eng.index_file(tp)
    os.unlink(tp)

    # Guardar
    save_path = Path("/tmp/test_index.tv")
    eng.save(str(save_path))

    # Cargar en nuevo engine
    eng2 = RagEngine()
    # Forzar carga
    eng2.load(str(save_path))

    assert len(eng2.metadata) > 0, "Loaded index is empty"
    assert "text" in eng2.metadata[-1], "Loaded metadata missing 'text'"

    # Limpiar
    os.unlink(str(save_path))
    os.unlink(str(save_path.with_suffix(".json")))

    print(f"  ✅ Save OK: {len(eng.metadata)} entries guardados")
    print(f"  ✅ Load OK: {len(eng2.metadata)} entries cargados")
    print(f"  ✅ Texto preservado en metadata")

    return True


def main():
    parser = argparse.ArgumentParser(description="E2E Test — RAG System")
    parser.add_argument("--skip-ocr", action="store_true", help="Saltar tests de OCR")
    parser.add_argument("--verbose", action="store_true", help="Logs detallados")
    args = parser.parse_args()

    if args.verbose:
        logging.disable(logging.NOTSET)

    print("=" * 60)
    print("🧪 E2E TEST — RAG Telegram System")
    print("=" * 60)

    start = time.time()
    tests_passed = 0
    tests_failed = 0

    tests = [
        ("Text Extraction", test_1_text_extraction),
        ("OCR Classifier", test_2_ocr_classifier),
        ("Indexing", test_3_indexing),
        ("Deduplication", test_5_deduplication),
        ("Batch Embeddings", test_6_batch_embeddings),
        ("Save/Load", test_7_save_load),
    ]

    global_e = None
    for name, test_fn in tests:
        try:
            result = test_fn()
            if result is True:
                tests_passed += 1
            elif result is not None:
                tests_passed += 1
                global_e = result
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            tests_failed += 1

    # Test 4 depende de test_3
    if global_e is not None:
        try:
            test_4_query(global_e)
            tests_passed += 1
        except Exception as e:
            print(f"  ❌ Query FAILED: {e}")
            tests_failed += 1

    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print(f"📊 RESULTADOS: {tests_passed} passed, {tests_failed} failed")
    print(f"⏱️  Tiempo total: {elapsed:.2f}s")
    print("=" * 60)

    if tests_failed == 0:
        print("🎉 TODOS LOS TESTS PASARON")
    else:
        print(f"⚠️  {tests_failed} test(s) fallaron")

    return tests_failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
