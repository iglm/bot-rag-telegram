#!/usr/bin/env python3
"""
turbovec_rag.py — RAG engine con TurboQuant para bot Telegram.
Indexa PDFs, responde preguntas usando retrieval + Hermes.

Mejoras implementadas:
  - MEJORA 1: chunking semántico con ai-chunking (RecursiveTextSplitter), fallback a chunk_text() clásico
  - MEJORA 3: ONNX Embeddings via optimum[onnxruntime] para inferencia más rápida en CPU
  - MEJORA 4: SqliteVectorStore como alternativa a TurboQuantIndex (engine="sqlite-vec")

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
import tempfile
from pathlib import Path

import numpy as np

# --- MEJORA 4: sqlite-vec como alternativa ---
try:
    import sqlite3
    import sqlite_vec
    HAS_SQLITE_VEC = True
except ImportError:
    HAS_SQLITE_VEC = False

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
DEFAULT_ENGINE = "turbovec"  # "turbovec" | "sqlite-vec"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# MEJORA 1: Chunking semántico con ai-chunking (fallback a chunk_text clásico)
# =============================================================================

def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP) -> list[str]:
    """
    Divide texto en chunks usando chunking semántico (ai-chunking).
    Fallback a chunking por palabras fijas si ai-chunking no está instalado.
    """
    # --- MEJORA 1: Intentar chunking semántico ---
    try:
        from ai_chunking import RecursiveTextSplitter
        splitter = RecursiveTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
        chunk_objects = splitter.chunk_text(text)
        # Los objetos Chunk tienen .text como campo principal
        chunks = [c.text for c in chunk_objects if c.text.strip()]
        if chunks:
            logger.info(f"✅ Chunking semántico (ai-chunking): {len(chunks)} chunks")
            return chunks
    except ImportError:
        logger.debug("ai-chunking no instalado, usando chunking por palabras fijas")
    except Exception as e:
        logger.warning(f"⚠️  chunking semántico falló ({e}), usando chunking por palabras fijas")

    # --- Fallback: chunking por palabras fijas (comportamiento original) ---
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    logger.debug(f"Chunking por palabras: {len(chunks)} chunks")
    return chunks


# =============================================================================
# MEJORA 3: ONNX Embeddings (más rápido en CPU)
# =============================================================================

def export_model_to_onnx(model_name: str = "BAAI/bge-m3", output_dir: str = None) -> str:
    """
    Exporta un modelo sentence-transformers a formato ONNX para inferencia más rápida.
    Requiere: optimum[onnxruntime]

    Args:
        model_name: Nombre del modelo HuggingFace
        output_dir: Directorio donde guardar el modelo ONNX (default: ~/.hermes/onnx_models/<model_name>)

    Returns:
        Ruta al directorio del modelo ONNX exportado, o cadena vacía si falla.
    """
    if output_dir is None:
        output_dir = str(Path.home() / ".hermes" / "onnx_models" / model_name.replace("/", "--"))

    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer
        import torch

        logger.info(f"📦 Exportando {model_name} a ONNX en {output_dir}...")

        # Cargar modelo PyTorch
        model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Guardar
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        logger.info(f"✅ Modelo ONNX exportado a {output_dir}")
        return output_dir
    except ImportError as e:
        logger.warning(f"⚠️  optimum/transformers no instalado para export ONNX: {e}")
    except Exception as e:
        logger.error(f"❌ Error exportando modelo ONNX: {e}")

    return ""


def _load_onnx_model(model_path: str):
    """
    Carga un modelo ONNX para embeddings.
    Retorna (model, tokenizer) o (None, None) si falla.
    """
    try:
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer

        model = ORTModelForFeatureExtraction.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info(f"✅ Modelo ONNX cargado desde {model_path}")
        return model, tokenizer
    except Exception as e:
        logger.warning(f"⚠️  No se pudo cargar modelo ONNX: {e}")
        return None, None


def _embed_with_onnx(texts: list[str], model, tokenizer, dim: int = 1024) -> np.ndarray:
    """
    Genera embeddings usando modelo ONNX.
    """
    import torch

    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling
    attention_mask = inputs["attention_mask"]
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    # Normalize
    embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)
    return embeddings.numpy().astype(np.float32)


# =============================================================================
# MEJORA 4: SqliteVectorStore — alternativa a TurboQuantIndex
# =============================================================================

class SqliteVectorStore:
    """
    Almacén vectorial usando sqlite-vec como alternativa a TurboQuantIndex.
    Para usar: engine="sqlite-vec" en RagEngine.
    """

    def __init__(self, dim: int = DEFAULT_DIM, db_path: str = None):
        self.dim = dim
        self.db_path = db_path or str(INDEX_DIR / "vectors.db")
        self.conn = None
        self.metadata: list[dict] = []
        self._init_db()

    def _init_db(self):
        """Inicializa la base de datos sqlite-vec."""
        if not HAS_SQLITE_VEC:
            raise ImportError("sqlite-vec no instalado. pip install sqlite-vec")

        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)

        # Crear tabla de vectores
        self.conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
            f"  chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,"
            f"  text_embedding FLOAT({self.dim})"
            f")"
        )
        self.conn.commit()

        # Cargar metadata existente
        self._load_metadata()

    def _load_metadata(self):
        """Carga metadata desde el archivo JSON."""
        meta_path = Path(self.db_path).with_suffix(".json")
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

    def _save_metadata(self):
        """Guarda metadata a JSON."""
        meta_path = Path(self.db_path).with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def add(self, embeddings: np.ndarray, metadata_batch: list[dict] = None):
        """
        Agrega embeddings al almacén.
        embeddings: numpy array de forma (N, dim)
        metadata_batch: lista de dicts con metadatos
        """
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        n = embeddings.shape[0]
        start_id = len(self.metadata)

        for i in range(n):
            emb = embeddings[i].tolist()
            row_id = start_id + i
            # Insertar vector
            self.conn.execute(
                "INSERT INTO vec_chunks(rowid, text_embedding) VALUES (?, ?)",
                (row_id, emb)
            )

        if metadata_batch:
            self.metadata.extend(metadata_batch)
            self._save_metadata()

        self.conn.commit()

    def search(self, query_embedding: np.ndarray, k: int = 5) -> tuple[list[list[float]], list[list[int]]]:
        """
        Busca los k vectores más cercanos.
        Retorna (scores, indices) en el mismo formato que TurboQuantIndex.search()
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_vec = query_embedding[0].tolist()
        n_queries = query_embedding.shape[0]

        all_scores = []
        all_indices = []

        for q in range(n_queries):
            # sqlite-vec retorna distancia, no similitud. Convertimos a score.
            cursor = self.conn.execute(
                "SELECT rowid, distance FROM vec_chunks WHERE text_embedding MATCH ? AND k = ?",
                (query_vec if q == 0 else query_embedding[q].tolist(), k)
            )
            results = cursor.fetchall()

            scores = []
            indices = []
            for rowid, distance in results:
                # Convertir distancia a score de similitud: score = 1 / (1 + distance)
                score = 1.0 / (1.0 + distance)
                scores.append(score)
                indices.append(rowid)

            all_scores.append(scores)
            all_indices.append(indices)

        return (all_scores, all_indices)

    def write(self, path: str):
        """Guarda el estado (compatible con TurboQuantIndex API)."""
        import shutil
        dest = Path(path)
        if self.db_path != str(dest):
            shutil.copy2(self.db_path, dest.with_suffix(".db"))
            shutil.copy2(Path(self.db_path).with_suffix(".json"), dest.with_suffix(".json"))

    @staticmethod
    def load(path: str):
        """Carga un almacén guardado."""
        db_path = Path(path).with_suffix(".db")
        if db_path.exists():
            store = SqliteVectorStore(db_path=str(db_path))
            store._load_metadata()
            return store
        raise FileNotFoundError(f"SqliteVectorStore no encontrado: {path}")

    def close(self):
        if self.conn:
            self.conn.close()


# =============================================================================
# RagEngine principal (con soporte para múltiples motores)
# =============================================================================

class RagEngine:
    def __init__(self, dim: int = DEFAULT_DIM, bit_width: int = DEFAULT_BIT_WIDTH,
                 model_name: str = "BAAI/bge-m3", engine: str = DEFAULT_ENGINE):
        """
        Args:
            dim: Dimensión de embeddings
            bit_width: Ancho de bits para TurboQuantIndex (solo si engine="turbovec")
            model_name: Nombre del modelo de embeddings
            engine: "turbovec" (TurboQuantIndex) o "sqlite-vec" (SqliteVectorStore)
        """
        self.dim = dim
        self.bit_width = bit_width
        self.engine_type = engine
        self.index = None
        self.metadata: list[dict] = []
        self.model = None
        self._onnx_model = None
        self._onnx_tokenizer = None
        self._model_name = model_name
        self._init_model()
        self._init_index()

    def _init_model(self):
        """Carga el modelo de embeddings, prefiriendo ONNX si está disponible."""
        # --- MEJORA 3: Intentar cargar modelo ONNX ---
        onnx_path = Path.home() / ".hermes" / "onnx_models" / self._model_name.replace("/", "--")
        if onnx_path.exists():
            onnx_model, onnx_tokenizer = _load_onnx_model(str(onnx_path))
            if onnx_model is not None:
                self._onnx_model = onnx_model
                self._onnx_tokenizer = onnx_tokenizer
                logger.info(f"✅ Usando modelo ONNX: {onnx_path}")
                return

        # Fallback a sentence-transformers (comportamiento original)
        from sentence_transformers import SentenceTransformer
        logger.info(f"Cargando modelo de embeddings: {self._model_name}...")
        self.model = SentenceTransformer(self._model_name)
        logger.info("✅ Modelo cargado (sentence-transformers)")

    def _init_index(self):
        """Inicializa o carga el índice según el motor elegido."""
        path = INDEX_FILE if self.engine_type == "turbovec" else INDEX_DIR / "vectors.db"

        if path.exists():
            self.load(INDEX_FILE if self.engine_type == "turbovec" else path)
        else:
            self._create_index()

    def _create_index(self):
        """Crea un nuevo índice."""
        if self.engine_type == "sqlite-vec":
            if not HAS_SQLITE_VEC:
                raise ImportError("sqlite-vec no instalado. pip install sqlite-vec")
            self.index = SqliteVectorStore(dim=self.dim, db_path=str(INDEX_DIR / "vectors.db"))
            logger.info(f"Nuevo SqliteVectorStore: dim={self.dim}")
        else:
            self.index = TurboQuantIndex(dim=self.dim, bit_width=self.bit_width)
            logger.info(f"Nuevo TurboQuantIndex: dim={self.dim}, bit_width={self.bit_width}")

    def _encode(self, texts: list[str]) -> np.ndarray:
        """
        Genera embeddings usando el modelo disponible.
        Prioriza ONNX si está cargado, fallback a sentence-transformers.
        """
        if self._onnx_model is not None:
            return _embed_with_onnx(texts, self._onnx_model, self._onnx_tokenizer, self.dim)
        else:
            return self.model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False).astype(np.float32)

    def text_to_embedding(self, text: str) -> np.ndarray:
        """Genera embedding para un solo texto."""
        return self._encode([text])[0]

    def index_file(self, file_path: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP):
        """Indexa un archivo (PDF, MD, JSON, TXT)."""
        logger.info(f"Indexando: {file_path}")
        text = extract_text_from_file(file_path)
        if not text.strip():
            logger.warning(f"⚠️  Texto vacío en {file_path}")
            return 0

        # --- MEJORA 1: chunking semántico ---
        chunks = chunk_text(text, chunk_size, overlap)
        if not chunks:
            return 0

        file_name = os.path.basename(file_path)

        # Batch embeddings
        logger.info(f"Generando {len(chunks)} embeddings en batch...")
        embeddings = self._encode(chunks)

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
        if self.engine_type == "sqlite-vec":
            self.index.add(embeddings, new_metadata)
        else:
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

        query_emb = self.text_to_embedding(question)
        query_array = query_emb.reshape(1, -1).astype(np.float32)

        if self.engine_type == "sqlite-vec":
            scores, indices = self.index.search(query_array, k=min(top_k, len(self.metadata)))
        else:
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
        if self.engine_type == "sqlite-vec":
            # SqliteVectorStore guarda automáticamente
            logger.info(f"💾 SqliteVectorStore persistente: {len(self.metadata)} chunks")
            return

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
        if self.engine_type == "sqlite-vec":
            db_path = Path(path) if path else INDEX_DIR / "vectors.db"
            if not db_path.exists():
                logger.error(f"❌ SqliteVectorStore no encontrado: {db_path}")
                return False
            self.index = SqliteVectorStore.load(str(db_path))
            self.metadata = self.index.metadata
            logger.info(f"📂 SqliteVectorStore cargado: {db_path} ({len(self.metadata)} chunks)")
            return True

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
        avg_chunk_len = sum(m["chunk_len"] for m in self.metadata) / total_chunks if total_chunks else 0

        print(f"📊 Estadísticas del índice:")
        print(f"   Motor: {self.engine_type}")
        print(f"   PDFs indexados: {len(pdfs)}")
        print(f"   Total chunks: {total_chunks}")
        print(f"   Chunk promedio: {avg_chunk_len:.0f} palabras")
        print(f"   Dimensión: {self.dim}")
        if self.engine_type == "turbovec":
            print(f"   Bit width: {self.bit_width}")
            print(f"   Compresión: {32 // self.bit_width}x vs float32")
        print(f"   PDFs:")
        for pdf in sorted(pdfs):
            count = sum(1 for m in self.metadata if m["pdf"] == pdf)
            print(f"     📄 {pdf}: {count} chunks")


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


def main():
    parser = argparse.ArgumentParser(description="RAG engine con TurboQuant")
    parser.add_argument("--engine", choices=["turbovec", "sqlite-vec"], default=DEFAULT_ENGINE,
                        help="Motor de almacén vectorial (default: turbovec)")
    parser.add_argument("--export-onnx", type=str, default=None, nargs="?",
                        const="BAAI/bge-m3",
                        help="Exportar modelo a ONNX (opcional: especificar nombre del modelo)")

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

    # --- MEJORA 3: Exportar modelo a ONNX ---
    if args.export_onnx:
        model_name = args.export_onnx
        output_path = export_model_to_onnx(model_name)
        if output_path:
            print(f"✅ Modelo ONNX exportado a: {output_path}")
        else:
            print(f"❌ Error exportando modelo ONNX")
        return

    if not args.command:
        parser.print_help()
        sys.exit(1)

    engine = RagEngine(engine=args.engine)

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
