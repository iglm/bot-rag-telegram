#!/usr/bin/env python3
"""
rag_eval.py — Evaluación automática de calidad del sistema RAG.
Métricas: relevancia de chunks, fidelidad de respuestas, recall@k.

MEJORA 3: open-rag-eval (Evaluación automática)
No bloqueante: todas las funciones tienen fallback si open-rag-eval no está instalado.

Uso:
  from rag_eval import evaluate_rag, compute_relevance, compute_faithfulness
  results = evaluate_rag(question, chunks, answer)
"""

import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# === Dependencias opcionales ===
HAS_OPEN_RAG_EVAL = False
try:
    # Intentar importar open-rag-eval
    from open_rag_eval.metrics import relevance, faithfulness
    from open_rag_eval.data_classes import RAGResult, ScoredRAGResult
    HAS_OPEN_RAG_EVAL = True
except ImportError:
    HAS_OPEN_RAG_EVAL = False

# === Configuración ===
EVAL_LOG_FILE = Path.home() / ".hermes" / "rag_eval_results.jsonl"


def compute_relevance(chunks: list[str], question: str) -> float:
    """
    Calcula la relevancia promedio de los chunks recuperados para una pregunta.

    Args:
        chunks: Lista de chunks de texto recuperados
        question: Pregunta del usuario

    Returns:
        Score de relevancia (0.0 - 1.0)
    """
    if not chunks or not question:
        return 0.0

    # Si open-rag-eval está disponible, usarlo
    if HAS_OPEN_RAG_EVAL:
        try:
            result = RAGResult(
                question=question,
                answer="",
                contexts=chunks,
            )
            scores = relevance(result)
            if scores:
                return sum(scores) / len(scores)
        except Exception as e:
            logger.debug(f"open-rag-eval relevance falló: {e}")

    # Fallback: relevancia basada en solapamiento de términos
    question_words = set(question.lower().split())
    if not question_words:
        return 0.0

    scores = []
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        if chunk_words:
            overlap = len(question_words & chunk_words)
            union = len(question_words | chunk_words)
            jaccard = overlap / union if union > 0 else 0
            scores.append(jaccard)

    return sum(scores) / len(scores) if scores else 0.0


def compute_faithfulness(answer: str, chunks: list[str]) -> float:
    """
    Calcula qué tan fiel es la respuesta a los chunks recuperados.

    Args:
        answer: Respuesta generada
        chunks: Chunks de contexto usados

    Returns:
        Score de fidelidad (0.0 - 1.0)
    """
    if not answer or not chunks:
        return 0.0

    # Si open-rag-eval está disponible
    if HAS_OPEN_RAG_EVAL:
        try:
            result = RAGResult(
                question="",
                answer=answer,
                contexts=chunks,
            )
            score = faithfulness(result)
            if score is not None:
                return score
        except Exception as e:
            logger.debug(f"open-rag-eval faithfulness falló: {e}")

    # Fallback: proporción de afirmaciones en answer que aparecen en chunks
    combined_context = " ".join(chunks).lower()
    answer_lower = answer.lower()

    # Extraer frases clave de la respuesta (oraciones)
    import re
    sentences = re.split(r'[.!?]+', answer_lower)
    supported = 0
    total = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:  # Ignorar frases muy cortas
            continue
        total += 1
        # Verificar si la oración contiene términos del contexto
        words = set(sentence.split()) - {
            "el", "la", "los", "las", "que", "y", "e", "o", "a", "de",
            "del", "en", "por", "para", "con", "un", "una", "es", "se",
            "su", "no", "lo", "como", "más", "pero", "sus", "le", "ya",
            "este", "entre", "porque", "cuando", "todo", "también", "fue",
            "era", "son", "han", "había", "hay", "muy", "sin", "sobre",
            "the", "and", "of", "to", "in", "is", "it", "that", "for",
            "are", "was", "with", "this", "have", "has", "from", "not",
        }
        if not words:
            continue
        # Si al menos algunas palabras están en los chunks
        match_count = sum(1 for w in words if w in combined_context)
        if match_count >= max(1, len(words) * 0.3):
            supported += 1

    return supported / total if total > 0 else 0.0


def compute_recall_at_k(relevant_chunks: list[str], retrieved_chunks: list[str], k: int = 5) -> float:
    """
    Calcula Recall@k: qué proporción de chunks relevantes fueron recuperados.

    Args:
        relevant_chunks: Chunks que son relevantes (ground truth)
        retrieved_chunks: Chunks recuperados por el sistema
        k: Número de chunks a considerar

    Returns:
        Recall@k (0.0 - 1.0)
    """
    if not relevant_chunks:
        return 0.0

    retrieved_at_k = set(retrieved_chunks[:k])
    relevant_set = set(relevant_chunks)

    if not retrieved_at_k or not relevant_set:
        return 0.0

    # Jaccard-like overlap
    overlap = len(retrieved_at_k & relevant_set)
    return overlap / len(relevant_set)


def evaluate_rag(
    question: str,
    answer: str,
    chunks: list[str],
    relevant_chunks: list[str] = None,
) -> dict:
    """
    Evaluación completa de una interacción RAG.

    Args:
        question: Pregunta del usuario
        answer: Respuesta generada
        chunks: Chunks recuperados
        relevant_chunks: Chunks ground-truth relevantes (opcional)

    Returns:
        dict con métricas: relevance, faithfulness, recall_at_k, ... 
    """
    result = {
        "question": question[:200],
        "answer_length": len(answer),
        "num_chunks": len(chunks),
    }

    # Relevancia
    try:
        result["relevance_score"] = compute_relevance(chunks, question)
    except Exception as e:
        logger.warning(f"Error computing relevance: {e}")
        result["relevance_score"] = 0.0

    # Fidelidad
    try:
        result["faithfulness_score"] = compute_faithfulness(answer, chunks)
    except Exception as e:
        logger.warning(f"Error computing faithfulness: {e}")
        result["faithfulness_score"] = 0.0

    # Recall@k
    if relevant_chunks:
        try:
            result["recall_at_5"] = compute_recall_at_k(relevant_chunks, chunks, k=5)
            result["recall_at_3"] = compute_recall_at_k(relevant_chunks, chunks, k=3)
        except Exception as e:
            logger.warning(f"Error computing recall: {e}")

    return result


def save_evaluation(result: dict):
    """Guarda resultado de evaluación a archivo JSONL."""
    EVAL_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(EVAL_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning(f"Error guardando evaluación: {e}")


def load_evaluations(limit: int = None) -> list[dict]:
    """Carga evaluaciones guardadas."""
    if not EVAL_LOG_FILE.exists():
        return []
    try:
        results = []
        with open(EVAL_LOG_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
        if limit:
            results = results[-limit:]
        return results
    except Exception as e:
        logger.warning(f"Error cargando evaluaciones: {e}")
        return []


def summary() -> dict:
    """Resumen de todas las evaluaciones guardadas."""
    results = load_evaluations()
    if not results:
        return {"total": 0}

    scores = {
        "relevance": [r.get("relevance_score", 0) for r in results if "relevance_score" in r],
        "faithfulness": [r.get("faithfulness_score", 0) for r in results if "faithfulness_score" in r],
    }

    return {
        "total_evaluations": len(results),
        "avg_relevance": sum(scores["relevance"]) / len(scores["relevance"]) if scores["relevance"] else 0,
        "avg_faithfulness": sum(scores["faithfulness"]) / len(scores["faithfulness"]) if scores["faithfulness"] else 0,
        "last_evaluation": results[-1] if results else None,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RAG Evaluation - Evalúa calidad del sistema")
    parser.add_argument("--question", type=str, help="Pregunta de prueba")
    parser.add_argument("--answer", type=str, help="Respuesta de prueba")
    parser.add_argument("--chunks", type=str, nargs="+", help="Chunks de contexto")
    parser.add_argument("--test", action="store_true", help="Ejecutar prueba simple")
    parser.add_argument("--summary", action="store_true", help="Mostrar resumen de evaluaciones")
    args = parser.parse_args()

    if args.summary:
        s = summary()
        print(json.dumps(s, indent=2, ensure_ascii=False))
        sys.exit(0)

    if args.test:
        print("🧪 Probando RAG Evaluation...")
        question = "¿Qué es el café colombiano?"
        chunks = [
            "El café colombiano es reconocido mundialmente por su calidad.",
            "Colombia produce café arábica de alta calidad en la región andina.",
            "El café de Colombia tiene denominación de origen protegida.",
        ]
        answer = "El café colombiano es reconocido mundialmente por su calidad. Se produce en la región andina y tiene denominación de origen protegida."

        result = evaluate_rag(question, answer, chunks, relevant_chunks=chunks)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\n📊 Relevance: {result['relevance_score']:.2f}")
        print(f"📊 Faithfulness: {result['faithfulness_score']:.2f}")

    if args.question and args.answer:
        chunks = args.chunks or []
        result = evaluate_rag(args.question, args.answer, chunks)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        save_evaluation(result)
