#!/usr/bin/env python3
"""
benchmark_embeddings.py — Evalúa embeddings actuales vs alternativas para español.
Usa mteb-es si está disponible para benchmarks estandarizados.

MEJORA 7: mteb-es (Benchmarks embeddings español)
No bloqueante: solo para diagnóstico.

Uso:
  python3 benchmark_embeddings.py          # Benchmark completo
  python3 benchmark_embeddings.py --quick  # Benchmark rápido
  python3 benchmark_embeddings.py --test   # Prueba simple
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# === Dependencias opcionales ===
HAS_MTEB = False
try:
    import mteb
    HAS_MTEB = True
except ImportError:
    HAS_MTEB = False

HAS_SENTENCE_TRANSFORMERS = False
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

HAS_ONNX = False
try:
    import onnxruntime
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

# === Textos de prueba en español ===
SPANISH_TEST_TEXTS = [
    "El café colombiano se cultiva en la región andina entre 1200 y 2000 metros de altitud.",
    "Colombia es el tercer productor mundial de café, después de Brasil y Vietnam.",
    "Las variedades de café más cultivadas en Colombia son Arábica Caturra, Colombia y Castillo.",
    "La Federación Nacional de Cafeteros de Colombia fue fundada en 1927.",
    "El café de Colombia tiene denominación de origen protegida desde 2007.",
    "El proceso de lavado del café colombiano produce granos de alta calidad.",
    "La zona cafetera colombiana comprende los departamentos de Caldas, Quindío y Risaralda.",
    "El café orgánico certificado representa el 5% de la producción nacional.",
    "Juan Valdez es el personaje emblemático del café colombiano desde 1958.",
    "La cosecha principal del café en Colombia ocurre entre octubre y diciembre.",
]

SPANISH_TEST_PAIRS = [
    ("El café colombiano es suave y bien balanceado", "Café de Colombia tiene acidez media y cuerpo suave"),
    ("El café se produce en montañas", "El café se cultiva en zonas montañosas"),
    ("La producción de café requiere clima templado", "El café necesita temperaturas entre 17 y 24 grados"),
    ("Colombia exporta café a Estados Unidos", "Estados Unidos es el principal comprador de café colombiano"),
    ("El café arábica es de mayor calidad", "Arábica es la variedad premium de café"),
]

SPANISH_QUERIES = [
    "¿Qué es el café colombiano?",
    "¿Dónde se cultiva el café en Colombia?",
    "Variedades de café en Colombia",
    "Historia del café colombiano",
    "Exportaciones de café de Colombia",
]


def load_bge_m3_model():
    """Carga el modelo bge-m3 (el usado actualmente en el sistema)."""
    if not HAS_SENTENCE_TRANSFORMERS:
        logger.warning("sentence-transformers no instalado")
        return None
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-m3")
        return model
    except Exception as e:
        logger.warning(f"Error cargando bge-m3: {e}")
        return None


def compute_embeddings(model, texts: list[str]) -> np.ndarray:
    """Genera embeddings para una lista de textos."""
    if model is None:
        return np.zeros((len(texts), 1024), dtype=np.float32)
    return model.encode(texts, normalize_embeddings=True, batch_size=16, show_progress_bar=False).astype(np.float32)


def benchmark_speed(model, texts: list[str], n_runs: int = 3) -> dict:
    """Mide velocidad de generación de embeddings."""
    times = []
    for _ in range(n_runs):
        start = time.time()
        compute_embeddings(model, texts)
        elapsed = time.time() - start
        times.append(elapsed / len(texts))
    return {
        "avg_time_per_text": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "texts_per_second": len(texts) / (sum(times) / len(times)),
    }


def benchmark_similarity(model, pairs: list[tuple[str, str]]) -> dict:
    """Evalúa calidad de similitud semántica entre pares de textos."""
    scores = []
    for t1, t2 in pairs:
        emb1 = compute_embeddings(model, [t1])[0]
        emb2 = compute_embeddings(model, [t2])[0]
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        scores.append(float(similarity))
    return {
        "avg_similarity": sum(scores) / len(scores) if scores else 0,
        "min_similarity": min(scores) if scores else 0,
        "max_similarity": max(scores) if scores else 0,
        "scores": scores,
    }


def benchmark_retrieval(model, corpus: list[str], queries: list[str], k: int = 3) -> dict:
    """Simula retrieval: cada query recupera del corpus."""
    if not corpus or not queries:
        return {"error": "No data"}

    corpus_embs = compute_embeddings(model, corpus)
    query_embs = compute_embeddings(model, queries)

    retrieval_times = []
    results = []

    for q_idx, q_emb in enumerate(query_embs):
        start = time.time()
        scores = np.dot(corpus_embs, q_emb)
        top_k = np.argsort(-scores)[:k]
        elapsed = time.time() - start
        retrieval_times.append(elapsed)
        results.append({
            "query": queries[q_idx],
            "top_k": [corpus[idx] for idx in top_k],
            "scores": [float(scores[idx]) for idx in top_k],
        })

    return {
        "avg_retrieval_time": sum(retrieval_times) / len(retrieval_times),
        "results": results,
    }


def full_benchmark() -> dict:
    """Benchmark completo del modelo actual."""
    logger.info("Cargando modelo bge-m3...")
    model = load_bge_m3_model()
    if model is None:
        return {"error": "No se pudo cargar modelo bge-m3"}

    model_name = "BAAI/bge-m3"
    dim = model.get_sentence_embedding_dimension() if hasattr(model, 'get_sentence_embedding_dimension') else 1024

    result = {
        "model": model_name,
        "dimension": dim,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Speed benchmark
    logger.info("Benchmark de velocidad...")
    result["speed"] = benchmark_speed(model, SPANISH_TEST_TEXTS)

    # Similarity benchmark
    logger.info("Benchmark de similitud semántica...")
    result["similarity"] = benchmark_similarity(model, SPANISH_TEST_PAIRS)

    # Retrieval benchmark
    logger.info("Benchmark de retrieval...")
    result["retrieval"] = benchmark_retrieval(model, SPANISH_TEST_TEXTS, SPANISH_QUERIES)

    # Memory usage (estimación)
    import psutil
    process = psutil.Process(os.getpid())
    result["memory_mb"] = process.memory_info().rss / 1024 / 1024

    return result


def quick_benchmark() -> dict:
    """Benchmark rápido (solo velocidad y similitud básica)."""
    model = load_bge_m3_model()
    if model is None:
        return {"error": "No se pudo cargar modelo"}

    result = {
        "model": "BAAI/bge-m3",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Solo 2 corridas para speed
    result["speed"] = benchmark_speed(model, SPANISH_TEST_TEXTS[:5], n_runs=2)
    result["similarity"] = benchmark_similarity(model, SPANISH_TEST_PAIRS[:3])

    return result


def run_mteb_benchmark(task_names: list[str] = None) -> dict:
    """Ejecuta benchmark MTEB estándar (si mteb está instalado)."""
    if not HAS_MTEB:
        return {
            "available": False,
            "error": "mteb no instalado. pip install mteb",
        }

    try:
        import mteb
        evaluation = mteb.MTEB(task_types=["Retrieval", "Clustering", "PairClassification"])

        if task_names:
            evaluation.tasks = [t for t in evaluation.tasks if t.description["name"] in task_names]

        results = evaluation.run("BAAI/bge-m3", verbosity=0)

        return {
            "available": True,
            "results": results,
            "tasks_completed": len(results),
        }
    except Exception as e:
        logger.warning(f"MTEB benchmark falló: {e}")
        return {
            "available": False,
            "error": str(e),
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark de embeddings para español")
    parser.add_argument("--quick", action="store_true", help="Benchmark rápido")
    parser.add_argument("--test", action="store_true", help="Prueba simple")
    parser.add_argument("--output", type=str, help="Archivo de salida JSON")
    parser.add_argument("--verbose", action="store_true", help="Logs detallados")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    if args.test:
        print("=" * 60)
        print("🧪 Benchmark de Embeddings — Prueba rápida")
        print("=" * 60)
        model = load_bge_m3_model()
        if model is None:
            print("❌ No se pudo cargar modelo")
            sys.exit(1)

        dim = model.get_sentence_embedding_dimension() if hasattr(model, 'get_sentence_embedding_dimension') else 1024
        print(f"📦 Modelo: BAAI/bge-m3 (dimensión: {dim})")

        emb = compute_embeddings(model, ["Texto de prueba en español"])
        print(f"📐 Embedding shape: {emb.shape}")
        print(f"📐 Norma: {np.linalg.norm(emb[0]):.4f}")

        sim = benchmark_similarity(model, SPANISH_TEST_PAIRS[:2])
        print(f"🔗 Similitud promedio: {sim['avg_similarity']:.4f}")

        speed = benchmark_speed(model, SPANISH_TEST_TEXTS[:3], n_runs=1)
        print(f"⚡ Velocidad: {speed['texts_per_second']:.1f} textos/segundo")
        sys.exit(0)

    if args.quick:
        print("⚡ Benchmark rápido...")
        result = quick_benchmark()
    else:
        print("📊 Benchmark completo de embeddings para español...")
        result = full_benchmark()

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"✅ Resultados guardados en: {args.output}")
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
