#!/usr/bin/env python3
"""
llm_router.py — Router inteligente de modelos LLM.
Elige el mejor modelo según el tipo de query:
  - Preguntas simples → modelo rápido/económico
  - Preguntas complejas → modelo potente
  - Preguntas en español → modelo multilingüe

MEJORA 2: LLMRouter (Routing inteligente de modelos)
Uso:
  from llm_router import select_model
  model = select_model("¿Qué es el café colombiano?")
"""

import logging
import os
import re

logger = logging.getLogger(__name__)

# === Configuración de modelos ===
# Rápidos/económicos (preguntas simples, factuales)
FAST_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemini-flash-1.5-8b:free",
    "mistralai/mistral-7b-instruct:free",
    "huggingfaceh4/zephyr-7b-beta:free",
]

# Potentes (preguntas complejas, análisis, razonamiento)
POWERFUL_MODELS = [
    "anthropic/claude-3.5-sonnet",
    "openai/gpt-4o",
    "google/gemini-pro-1.5",
    "mistralai/mistral-large",
    "qwen/qwen-2.5-72b-instruct",
]

# Multilingües (español, otros idiomas no-inglés)
MULTILINGUAL_MODELS = [
    "qwen/qwen-2.5-72b-instruct",
    "google/gemini-pro-1.5",
    "mistralai/mistral-medium",
    "nvidia/nemotron-3-super-120b-a12b:free",
]

# === Palabras clave para clasificación ===
SIMPLE_QUERY_PATTERNS = [
    r"qu[eé] es",
    r"qu[eé] significa",
    r"definici[oó]n",
    r"cu[aá]ndo",
    r"d[oó]nde",
    r"qui[eé]n",
    r"cu[aá]l es",
    r"cu[aá]nto",
    r"lista",
    r"enumera",
    r"ejemplos? de",
    r"tipos? de",
    r"fecha",
    r"horario",
    r"precio",
    r"costo",
]

COMPLEX_QUERY_PATTERNS = [
    r"compara",
    r"diferencia",
    r"analiza",
    r"explica detalladamente",
    r"por qu[eé]",
    r"c[oó]mo funciona",
    r"razona",
    r"argumenta",
    r"resume",
    r"sintetiza",
    r"relación entre",
    r"impacto de",
    r"consecuencias",
    r"implicaciones",
    r"beneficios y",
    r"ventajas y desventajas",
    r"pros y contras",
    r"causa",
    r"efecto",
    r"tendencia",
    r"evolución",
    r"predice",
    r"pronostica",
]

# Palabras españolas comunes para detección de idioma
SPANISH_INDICATORS = [
    "el", "la", "los", "las", "que", "es", "por", "para", "con", "sin",
    "del", "como", "más", "pero", "tiene", "este", "esta", "estos",
    "entre", "sobre", "según", "durante", "mediante", "gracias",
    "cual", "cuales", "donde", "cuando", "quien", "porque",
    "español", "española", "colombia", "méxico", "argentina", "chile",
    "perú", "venezuela", "ecuador", "guatemala", "cuba", "bolivia",
    "república", "dominicana", "honduras", "paraguay", "salvador",
    "nicaragua", "costa", "rica", "panamá", "uruguay", "puerto", "rico",
    "también", "además", "entonces", "siempre", "nunca", "ambos",
    "cualquier", "durante", "embargo", "favor", "gracias",
]

# === Modelo por defecto (fallback) ===
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")

# === Cache simple de decisión ===
_router_cache: dict = {}


def _detect_spanish(query: str) -> bool:
    """Detecta si la query está principalmente en español."""
    if not query.strip():
        return False
    words = re.findall(r'\w+', query.lower())
    if not words:
        return False
    spanish_count = sum(1 for w in words if w in SPANISH_INDICATORS)
    return spanish_count >= 3


def _is_simple_query(query: str) -> bool:
    """Determina si la query es simple (búsqueda factual, definición, etc)."""
    query_lower = query.lower().strip()
    for pattern in SIMPLE_QUERY_PATTERNS:
        if re.search(pattern, query_lower):
            return True
    # Queries cortas son probablemente simples
    if len(query.split()) <= 4:
        return True
    return False


def _is_complex_query(query: str) -> bool:
    """Determina si la query es compleja (requiere razonamiento)."""
    query_lower = query.lower().strip()
    for pattern in COMPLEX_QUERY_PATTERNS:
        if re.search(pattern, query_lower):
            return True
    # Queries largas con múltiples partes
    word_count = len(query.split())
    if word_count >= 20:
        return True
    return False


def select_model(query: str, default: str = None) -> str:
    """
    Selecciona el mejor modelo LLM según el tipo de query.

    Args:
        query: La pregunta del usuario
        default: Modelo por defecto si no se puede determinar

    Returns:
        Nombre del modelo a usar
    """
    if default is None:
        default = DEFAULT_MODEL

    if not query or not query.strip():
        return default

    # Cache hit
    cache_key = query.strip().lower()[:100]
    if cache_key in _router_cache:
        return _router_cache[cache_key]

    # Detectar idioma
    is_spanish = _detect_spanish(query)

    # Clasificar complejidad
    is_simple = _is_simple_query(query)
    is_complex = _is_complex_query(query)

    # Decidir modelo
    if is_spanish:
        # Para español, preferir modelos multilingües
        if is_complex:
            # Español + complejo → mejor modelo multilingüe disponible
            model = MULTILINGUAL_MODELS[0]
            _log_decision(query, model, "multilingual+complex", is_spanish)
        else:
            # Español simple → modelo rápido multilingüe
            model = MULTILINGUAL_MODELS[-1] if len(MULTILINGUAL_MODELS) > 1 else DEFAULT_MODEL
            _log_decision(query, model, "multilingual+simple", is_spanish)
    elif is_complex:
        # Complejo (inglés) → modelo potente
        model = POWERFUL_MODELS[0]
        _log_decision(query, model, "powerful", is_spanish)
    elif is_simple:
        # Simple (inglés) → modelo rápido
        model = FAST_MODELS[0]
        _log_decision(query, model, "fast", is_spanish)
    else:
        # Neutro → modelo por defecto
        model = default
        _log_decision(query, model, "default", is_spanish)

    # Cache
    _router_cache[cache_key] = model
    return model


def _log_decision(query: str, model: str, category: str, is_spanish: bool):
    """Log interno de la decisión del router."""
    logger.debug(
        f"LLMRouter: [{category}] {'[ES]' if is_spanish else '[EN]'} "
        f"query={query[:60]}... → model={model}"
    )


def classify_query(query: str) -> dict:
    """
    Clasifica una query y retorna metadatos sobre la decisión.

    Args:
        query: La pregunta del usuario

    Returns:
        dict con: model, category, is_spanish, reason
    """
    is_spanish = _detect_spanish(query)
    is_simple = _is_simple_query(query)
    is_complex = _is_complex_query(query)

    if is_spanish and is_complex:
        category = "multilingual+complex"
    elif is_spanish:
        category = "multilingual+simple"
    elif is_complex:
        category = "powerful"
    elif is_simple:
        category = "fast"
    else:
        category = "default"

    model = select_model(query)

    reason_map = {
        "multilingual+complex": "Query en español que requiere razonamiento",
        "multilingual+simple": "Query en español, pregunta simple",
        "powerful": "Query compleja que requiere modelo potente",
        "fast": "Query simple que puede usar modelo rápido",
        "default": "Query neutra, usando modelo por defecto",
    }

    return {
        "model": model,
        "category": category,
        "is_spanish": is_spanish,
        "is_simple": is_simple,
        "is_complex": is_complex,
        "reason": reason_map.get(category, "Decisión por defecto"),
    }


def clear_cache():
    """Limpia el cache interno del router."""
    _router_cache.clear()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LLM Router - Selecciona modelo según query")
    parser.add_argument("query", nargs="?", help="Query de prueba")
    parser.add_argument("--verbose", action="store_true", help="Log detallado")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    test_queries = [
        args.query,
        "¿Qué es el café colombiano?",
        "¿Cuál es la diferencia entre café arábica y robusta?",
        "Lista de variedades de café",
        "Analiza el impacto del cambio climático en la producción de café en Colombia y sus consecuencias económicas para los pequeños productores",
        "Explain how transformer models work",
        "Hello world",
    ] if not args.query else [args.query]

    print("=" * 60)
    print("🧠 LLM Router — Prueba de clasificación")
    print("=" * 60)
    for q in test_queries:
        if not q:
            continue
        result = classify_query(q)
        print(f"\n📝 Query: {q[:80]}")
        print(f"   Modelo: {result['model']}")
        print(f"   Categoría: {result['category']}")
        print(f"   Español: {result['is_spanish']}")
        print(f"   Razón: {result['reason']}")
