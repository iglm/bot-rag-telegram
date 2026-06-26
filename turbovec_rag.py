#!/usr/bin/env python3
"""
turbovec_rag.py — RAG engine con TurboQuant para bot Telegram.
Indexa PDFs, responde preguntas usando retrieval + Hermes.

Uso:
  python3 turbovec_rag.py index <pdf_path> [--chunk-size 500] [--overlap 100]
  python3 turbovec_rag.py index-dir <directorio_pdfs> [--chunk-size 500]
  python3 turbovec_rag.py query <pregunta> [--top-k 5]
  python3 turbovec_rag.py save <path.tv>
  python3 turbovec_rag.py load <path.tv>
  python3 turbovec_rag.py stats
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np

try:
    from turbovec import TurboQuantIndex
except ImportError:
    print("❌ turbovec no instalado. Ejecuta: pip install turbovec")
    sys.exit(1)

try:
    import pdfplumber
except ImportError:
    print("❌ pdfplumber no instalado. Ejecuta: pip install pdfplumber")
    sys.exit(1)

# Config
DEFAULT_DIM = 1024
DEFAULT_BIT_WIDTH = 4
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 100
INDEX_DIR = Path.home() / ".hermes" / "turbovec_index"
INDEX_FILE = INDEX_DIR / "index.tv"
META_FILE = INDEX_DIR / "metadata.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def text_to_embedding(text: str, model, dim: int = 1024) -> np.ndarray:
    """Genera embedding usando sentence-transformers."""
    emb = model.encode(text, normalize_embeddings=True)
    return emb.astype(np.float32)


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> list[str]:
    """Divide texto en chunks con overlap."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extrae texto de un PDF."""
    text_parts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
    except Exception as e:
        logger.error(f"Error extrayendo {pdf_path}: {e}")
    return "\n".join(text_parts)


def extract_text_from_file(file_path: str) -> str:
    """Extrae texto de PDF, MD, JSON o TXT."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    try:
        if suffix == ".pdf":
            return extract_text_from_pdf(file_path)
        elif suffix in (".md", ".txt"):
            return path.read_text(encoding="utf-8")
        elif suffix == ".json":
            # JSON — extraer valores de texto
            data = json.loads(path.read_text(encoding="utf-8"))
            return _extract_text_from_json(data)
        else:
            logger.warning(f"Formato no soportado: {suffix}")
            return ""
    except Exception as e:
        logger.error(f"Error extrayendo {file_path}: {e}")
        return ""


def _extract_text_from_json(data, depth=0) -> str:
    """Extrae recursivamente valores de texto de un JSON."""
    if depth > 10:
        return ""
    parts = []
    if isinstance(data, str):
        parts.append(data)
    elif isinstance(data, dict):
        for v in data.values():
            parts.append(_extract_text_from_json(v, depth + 1))
    elif isinstance(data, list):
        for item in data:
            parts.append(_extract_text_from_json(item, depth + 1))
    return "\n".join(parts)


class RagEngine:
    def __init__(self, dim: int = DEFAULT_DIM, bit_width: int = DEFAULT_BIT_WIDTH, model_name: str = "BAAI/bge-m3"):
        self.dim = dim
        self.bit_width = bit_width
        self.index: TurboQuantIndex = None
        self.metadata: list[dict] = []
        self.model = None
        self._model_name = model_name
        self._init_model()
        self._init_index()

    def _init_model(self):
        """Carga el modelo de embeddings."""
        from sentence_transformers import SentenceTransformer
        logger.info(f"Cargando modelo de embeddings: {self._model_name}...")
        self.model = SentenceTransformer(self._model_name)
        logger.info("✅ Modelo cargado")

    def _init_index(self):
        """Inicializa o carga el índice."""
        if INDEX_FILE.exists():
            self.load(INDEX_FILE)
        else:
            self.index = TurboQuantIndex(dim=self.dim, bit_width=self.bit_width)
            logger.info(f"Índice nuevo creado: dim={self.dim}, bit_width={self.bit_width}")

    def index_file(self, file_path: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP):
        """Indexa un archivo (PDF, MD, JSON, TXT)."""
        logger.info(f"Indexando: {file_path}")
        text = extract_text_from_file(file_path)
        if not text.strip():
            logger.warning(f"⚠️  Texto vacío en {file_path}")
            return 0

        chunks = chunk_text(text, chunk_size, overlap)
        if not chunks:
            return 0

        file_name = os.path.basename(file_path)

        # Batch embeddings (PERF-4): mucho más rápido que uno por uno
        logger.info(f"Generando {len(chunks)} embeddings en batch...")
        embeddings = self.model.encode(chunks, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        embeddings = embeddings.astype(np.float32)

        new_metadata = []
        for i, chunk in enumerate(chunks):
            new_metadata.append({
                "pdf": file_name,
                "chunk_id": i,
                "text": chunk,
                "text_preview": chunk[:200],
                "chunk_len": len(chunk),
            })

        # Agregar al índice
        self.index.add(embeddings)
        self.metadata.extend(new_metadata)

        logger.info(f"✅ {len(chunks)} chunks indexados de {file_name}")
        return len(chunks)

    def index_pdf(self, pdf_path: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP):
        """Indexa un PDF (wrapper backward-compatible)."""
        return self.index_file(pdf_path, chunk_size, overlap)

    def index_directory(self, dir_path: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP):
        """Indexa todos los archivos soportados de un directorio."""
        dir_path = Path(dir_path)
        supported = ("*.pdf", "*.md", "*.json", "*.txt")
        files = []
        for ext in supported:
            files.extend(sorted(dir_path.glob(ext)))
        if not files:
            logger.warning(f"⚠️  No se encontraron archivos en {dir_path}")
            return 0

        total = 0
        for f in files:
            count = self.index_file(str(f), chunk_size, overlap)
            total += count

        logger.info(f"📚 Total: {total} chunks de {len(files)} archivos")
        return total

    def query(self, question: str, top_k: int = 5) -> list[dict]:
        """Busca chunks relevantes para una pregunta."""
        if not self.metadata:
            logger.warning("⚠️  Índice vacío. Indexa PDFs primero.")
            return []

        query_emb = text_to_embedding(question, self.model, self.dim)
        query_array = query_emb.reshape(1, -1).astype(np.float32)

        scores, indices = self.index.search(query_array, k=min(top_k, len(self.metadata)))
        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.metadata):
                meta = self.metadata[idx]
                results.append({
                    "score": float(score),
                    "pdf": meta["pdf"],
                    "chunk_id": meta["chunk_id"],
                    "text_preview": meta["text_preview"],
                    "full_text": self._get_chunk_text(idx)
                })

        return results

    def _get_chunk_text(self, idx: int) -> str:
        """Recupera el texto completo de un chunk por índice."""
        if idx < len(self.metadata):
            return self.metadata[idx].get("text", self.metadata[idx]["text_preview"])
        return ""

    def save(self, path: str = None):
        """Guarda el índice y metadata."""
        path = Path(path) if path else INDEX_FILE
        path.parent.mkdir(parents=True, exist_ok=True)

        # Guardar índice turbovec
        self.index.write(str(path))

        # Guardar metadata
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"💾 Índice guardado: {path} ({len(self.metadata)} chunks)")

    def load(self, path: str = None):
        """Carga el índice y metadata."""
        path = Path(path) if path else INDEX_FILE

        if not path.exists():
            logger.error(f"❌ Índice no encontrado: {path}")
            return False

        self.index = TurboQuantIndex.load(str(path))

        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        logger.info(f"📂 Índice cargado: {path} ({len(self.metadata)} chunks)")
        return True

    def stats(self):
        """Estadísticas del índice."""
        if not self.metadata:
            print("📊 Índice vacío")
            return

        pdfs = set(m["pdf"] for m in self.metadata)
        total_chunks = len(self.metadata)
        avg_chunk_len = sum(m["chunk_len"] for m in self.metadata) / total_chunks

        print(f"📊 Estadísticas del índice:")
        print(f"   PDFs indexados: {len(pdfs)}")
        print(f"   Total chunks: {total_chunks}")
        print(f"   Chunk promedio: {avg_chunk_len:.0f} palabras")
        print(f"   Dimensión: {self.dim}")
        print(f"   Bit width: {self.bit_width}")
        print(f"   Compresión: {32 // self.bit_width}x vs float32")
        print(f"   PDFs:")
        for pdf in sorted(pdfs):
            count = sum(1 for m in self.metadata if m["pdf"] == pdf)
            print(f"     📄 {pdf}: {count} chunks")


def main():
    parser = argparse.ArgumentParser(description="RAG engine con TurboQuant")
    subparsers = parser.add_subparsers(dest="command", help="Comando")

    # Index
    idx_parser = subparsers.add_parser("index", help="Indexar un PDF")
    idx_parser.add_argument("pdf", help="Ruta al PDF")
    idx_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    idx_parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)

    # Index directory
    dir_parser = subparsers.add_parser("index-dir", help="Indexar directorio de PDFs")
    dir_parser.add_argument("directorio", help="Directorio con PDFs")
    dir_parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    dir_parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)

    # Query
    q_parser = subparsers.add_parser("query", help="Buscar en el índice")
    q_parser.add_argument("pregunta", help="Pregunta o texto de búsqueda")
    q_parser.add_argument("--top-k", type=int, default=5)

    # Save
    save_parser = subparsers.add_parser("save", help="Guardar índice")
    save_parser.add_argument("path", nargs="?", default=None, help="Ruta destino")

    # Load
    load_parser = subparsers.add_parser("load", help="Cargar índice")
    load_parser.add_argument("path", help="Ruta del índice")

    # Stats
    subparsers.add_parser("stats", help="Estadísticas del índice")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    engine = RagEngine()

    if args.command == "index":
        engine.index_pdf(args.pdf, args.chunk_size, args.overlap)
        engine.save()

    elif args.command == "index-dir":
        engine.index_directory(args.directorio, args.chunk_size, args.overlap)
        engine.save()

    elif args.command == "query":
        results = engine.query(args.pregunta, args.top_k)
        if not results:
            print("No se encontraron resultados.")
        else:
            print(f"\n🔍 Top-{len(results)} resultados:\n")
            for i, r in enumerate(results, 1):
                print(f"  {i}. [{r['pdf']}] (score: {r['score']:.2f})")
                print(f"     {r['text_preview'][:150]}...")
                print()

    elif args.command == "save":
        engine.save(args.path)

    elif args.command == "load":
        engine.load(args.path)

    elif args.command == "stats":
        engine.stats()


if __name__ == "__main__":
    main()
