# 🤖 Telegram RAG Bot — "Frankenstein"

> Un sistema RAG (Retrieval-Augmented Generation) para Telegram, armado con las mejores piezas open source del mercado, funcional 100% local y gratis.

## 🧟 ¿Por qué "Frankenstein"?

Porque no es un framework monolítico, sino un **monstruo armado con piezas de distintos fabricantes**, cada una especializada en su tarea, trabajando juntas como un sistema coherente:

| Pieza | Origen | Función |
|-------|--------|---------|
| TurboQuant | Google Research (Rust) | Compresión 8x de vectores |
| bge-m3 | BAAI (HuggingFace) | Embeddings multilingües |
| ai-chunking | nexla-opensource | División semántica de texto |
| MinerU | OpenDataLab | Parser PDF→MD (5x más rápido) |
| PDF Oxide | fedoseev (Rust) | Parser PDF ultra rápido |
| PaddleOCR | Baidu | OCR para español |
| GPTCache | Zilliz | Cache de embeddings |
| LLMRouter | UIUC | Routing inteligente de modelos |
| LightRAG | HKUDS | RAG con grafos semánticos |
| RedisVL | Redis | Memoria conversacional |
| open-rag-eval | Vectara | Evaluación automática |
| TruLens | Truera | Tracking de calidad |
| sqlite-vec | sqliteai | Búsqueda vectorial SQLite |
| aiogram | Framework estándar | Bot Telegram |

---

## 🎯 Características

- **Indexación automática** de PDFs, MDs, JSONs y TXTs en grupos Telegram
- **4 motores OCR** que se eligen automáticamente según el tipo de documento
- **Embeddings profesionales** con BAAI/bge-m3 (multilingüe, 1024-dim)
- **Compresión 8x** con turbovec 4-bit (Google TurboQuant)
- **Chunking semántico** que divide por ideas, no por palabras
- **Cache de embeddings** (no re-genera documentos ya indexados)
- **Routing inteligente** que elige el mejor modelo según la consulta
- **Memoria conversacional** entre sesiones
- **Evaluación de calidad** automática del RAG
- **$0/mes** — todo corre en VPS sin GPU

## 📊 Arquitectura

```
📄 Documento subido (PDF/MD/JSON/TXT)
       │
       ▼
┌──────────────────────────────────────────────┐
│ 🤖 CLASIFICADOR OCR (ocr_classifier.py)      │
│                                              │
│  ¿Digital complejo? → MinerU (5x rápido)    │
│  ¿Digital simple?   → PDF Oxide (más rápido) │
│  ¿Escaneado?        → PaddleOCR (español)   │
│  ¿Muy complejo?     → docAI pipeline         │
│  ¿Fallback?         → pymupdf4llm           │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│ 🧩 CHUNKING SEMÁNTICO (ai-chunking)          │
│                                              │
│  Divide por ideas/párrafos, no por palabras   │
│  Fallback: chunking clásico por palabras      │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│ 🚀 EMBEDDINGS (turbovec_rag.py)             │
│                                              │
│  Modelo: BAAI/bge-m3 (1024-dim)              │
│  Cache: GPTCache (SHA256, TTL 24h)           │
│  ONNX: 2x más rápido en CPU (opcional)       │
│  Fallback: PyTorch estándar                  │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│ 🗄️ VECTOR STORE (elige uno)                  │
│                                              │
│  TurboQuantIndex (turbovec, 8x compresión)   │
│  SqliteVectorStore (sqlite-vec, alternativa) │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│ 🔍 RETRIEVAL + MEMORIA                       │
│                                              │
│  Query → Embedding → Top-k chunks            │
│  Contexto COMPLETO (no preview 200 chars)    │
│  Memoria conversacional (RedisVL)            │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│ 🤖 LLM ROUTING (llm_router.py)               │
│                                              │
│  Simple → modelo rápido/económico            │
│  Compleja → modelo potente                   │
│  Español → modelo multilingüe                │
│  Fallback: OPENROUTER_MODEL                  │
└──────────────┬───────────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────────┐
│ 📤 RESPUESTA → Telegram                      │
│                                              │
│  Con contexto completo + score + fuente      │
└──────────────────────────────────────────────┘
```

## 🚀 Instalación

### Requisitos

- Python 3.11+
- VPS Linux (2 CPU, 4GB RAM mínimo)
- 5GB disco libre (~2GB para modelos)
- Tesseract OCR (`apt install tesseract-ocr tesseract-ocr-spa`)

### Pasos

```bash
# 1. Clonar
git clone https://github.com/iglm/bot-rag-telegram.git
cd bot-rag-telegram

# 2. Crear venv
python3 -m venv venv
source venv/bin/activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env
# Editar .env con tus credenciales

# 5. Iniciar
python3 bot_documentos_indexados.py
```

### Variables de Entorno

```env
BOT_TOKEN_DOCS=tu_token_de_botfather
OPENROUTER_API_KEY=tu_key_de_openrouter
OPENROUTER_MODEL=nvidia/nemotron-3-super-120b-a12b:free
```

## 📋 Uso en Telegram

1. **Crea un grupo** en Telegram
2. **Agrega el bot** como administrador
3. **Sube documentos** (PDF, MD, JSON, TXT) — indexación automática
4. **Haz preguntas** directamente en el grupo

| Comando | Acción |
|---------|--------|
| Subir archivo | Indexación automática |
| `/indexar` | Re-indexar todos los documentos del chat |
| `/estado` | Ver estadísticas del índice |
| `/buscar <pregunta>` | Búsqueda con respuesta IA |
| `/memoria` | Ver historial conversacional |
| `/clear_memoria` | Limpiar memoria |
| `/help` | Ver ayuda |
| Pregunta directa | Búsqueda automática |

## 🔧 Configuración OCR

El sistema elige automáticamente el mejor motor según el tipo de PDF:

| Tipo de PDF | Motor | Por qué |
|-------------|-------|---------|
| Digital complejo (tablas, columnas) | MinerU | 5x más rápido, maneja layouts |
| Digital simple | PDF Oxide | Ultra rápido, 100% precisión |
| Escaneado español | PaddleOCR | Mejor soporte español |
| Muy complejo | docAI pipeline | Análisis de layout + OCR |
| Fallback | pymupdf4llm | Siempre funciona |

## 📊 Rendimiento

| Métrica | Valor |
|---------|-------|
| Memoria RAM | ~150MB (bot) + ~2GB (modelo embeddings) |
| Indexación PDF digital | ~1-2s (MinerU) |
| Indexación PDF escaneado | ~5-10s (PaddleOCR) |
| Búsqueda | <100ms para 100K chunks |
| Compresión índice | 8x vs float32 |
| Disco para 100 PDFs | ~30MB (MD) vs ~3GB (PDFs) |
| Costo | $0/mes |

## 📁 Estructura del Proyecto

```
bot-rag-telegram/
├── bot_documentos_indexados.py   # Bot Telegram (aiogram)
├── turbovec_rag.py              # Motor RAG (embeddings + turbovec + cache)
├── ocr_classifier.py            # Clasificador inteligente OCR
├── ocr_converter.py             # Conversor PDF → MD (4 motores)
├── light_rag_engine.py          # RAG con grafos semánticos
├── llm_router.py                # Router inteligente de modelos LLM
├── rag_eval.py                  # Evaluación automática de RAG
├── conversation_memory.py       # Memoria conversacional
├── benchmark_embeddings.py      # Benchmarks embeddings español
├── test_e2e.py                  # Tests end-to-end (12+ tests)
├── requirements.txt             # Dependencias
├── install.sh                   # Script de instalación
├── bot-docs-indexados.service   # Servicio systemd
├── .env.example                 # Plantilla de variables
└── README.md                    # Este archivo
```

## 🔒 Seguridad

- API keys en `.env` (no en código ni systemd)
- `.env` con permisos 600
- Bot solo responde en el grupo autorizado
- Sin telemetría ni analytics
- No se guardan PDFs originales (solo MD convertido)

## 🧪 Tests

```bash
python3 test_e2e.py
```

Incluye: extracción de texto, clasificación OCR, indexación, embeddings en batch, save/load, retrieval, chunking semántico, y más.

## 📝 Licencia

MIT

---

**Creado por:** Lucas Mateo Tabares Franco
**Créditos:** Lucas Mateo Tabares Franco + Ing. Jhoan Sebastian Bustamante Montes
