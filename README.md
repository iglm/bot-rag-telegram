# 🤖 Telegram RAG Bot — NotebookLM-like con TurboQuant

Bot Telegram para indexar documentos y responder preguntas usando RAG (Retrieval-Augmented Generation) 100% local y gratis.

## 🎯 Características

- **Indexación automática** de PDFs, MDs, JSONs y TXTs en grupos Telegram
- **Conversión OCR inteligente**: detecta tipo de PDF y elige la mejor herramienta
- **Embeddings profesionales** con BAAI/bge-m3 (multilingüe, 1024-dim)
- **Compresión 8x** con turbovec 4-bit (Google TurboQuant)
- **Respuestas IA** via OpenRouter (modelos gratuitos)
- **$0/mes** — todo corre en VPS sin GPU

## 📊 Arquitectura

```
📄 Documento subido (PDF/MD/JSON/TXT)
       ↓
🤖 Agente clasificador OCR (analiza contenido)
       ↓
┌─────────────────────────────────────────┐
│ PDF digital    → pymupdf4llM (rápido)   │
│ Escaneado      → OCRmyPDF (Tesseract)   │
│ Largo/complejo → Unlimited-OCR (Baidu)  │
│ MD/JSON/TXT    → indexación directa      │
└─────────────────────────────────────────┘
       ↓
🗄️ TurboQuant Index (turbovec 4-bit, 1024-dim)
       ↓
🔍 Retrieval top-k → contexto
       ↓
🧠 OpenRouter Free LLM → respuesta
       ↓
📤 Telegram reply
```

## 🛠️ Stack

| Componente | Tecnología | Costo |
|------------|-----------|-------|
| LLM | OpenRouter (nemotron-3, qwen3-coder) | Gratis |
| Embeddings | sentence-transformers + BAAI/bge-m3 | Gratis |
| Vector Index | turbovec (Google TurboQuant 4-bit) | Gratis |
| OCR Digital | pymupdf4llm | Gratis |
| OCR Escaneado | ocrmypdf (Tesseract) | Gratis |
| OCR Complejo | Unlimited-OCR (PaddleOCR) | Gratis |
| Bot | aiogram 3.x | Gratis |

## 🚀 Instalación

### Requisitos

- Python 3.11+
- VPS Linux (2 CPU, 4GB RAM mínimo)
- 5GB disco libre (~2GB para modelos)

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
export BOT_TOKEN_DOCS="tu_token_de_botfather"
export OPENROUTER_API_KEY="tu_key_de_openrouter"
# Opcional: modelo por defecto
export OPENROUTER_MODEL="nvidia/nemotron-3-super-120b-a12b:free"

# 5. Iniciar
python3 bot_documentos_indexados.py
```

### Variables de Entorno

| Variable | Requerido | Descripción |
|----------|-----------|-------------|
| `BOT_TOKEN_DOCS` | ✅ | Token de BotFather |
| `OPENROUTER_API_KEY` | ✅ | API key de OpenRouter |
| `OPENROUTER_MODEL` | Opcional | Modelo LLM (default: nemotron-3-super) |

## 📋 Uso

### En el grupo Telegram

| Comando | Acción |
|---------|--------|
| Subir PDF/MD/JSON/TXT | Indexación automática |
| `/indexar` | Re-indexar documentos del chat |
| `/estado` | Ver estadísticas del índice |
| `/buscar <pregunta>` | Búsqueda con respuesta IA |
| `/help` | Ver ayuda |
| Pregunta directa | Búsqueda automática |

### Ejemplo

```
Usuario: ¿Cuál es el costo de fertilización por hectárea?

Bot: 🔍 Buscando...

Bot: Según los documentos indexados, la fertilización representa 
     el 19% de los costos de producción de café. Para una finca 
     de 20ha en Manizales, el costo promedio es de $2.8M COP/ha 
     (FEPCafé 2024).
```

## 📁 Estructura

```
bot-rag-telegram/
├── bot_documentos_indexados.py   # Bot Telegram (aiogram)
├── turbovec_rag.py              # Motor RAG (embeddings + turbovec)
├── ocr_classifier.py            # Agente clasificador OCR
├── ocr_converter.py             # Conversor PDF → MD
├── requirements.txt             # Dependencias
├── install.sh                   # Script de instalación
├── bot-docs-indexados.service   # Servicio systemd
└── README.md                    # Este archivo
```

## 🔧 Configuración OCR

El agente clasificador analiza cada PDF y decide automáticamente:

| Tipo de PDF | Herramienta | Cuando |
|-------------|-------------|--------|
| Digital (texto seleccionable) | pymupdf4llm | >30% páginas con texto |
| Escaneado simple | OCRmyPDF | <100 páginas, >50% sin texto |
| Largo/complejo | Unlimited-OCR | >50 páginas escaneadas |
| Mixto | pymupdf4llm | Texto + imágenes |

## 📊 Rendimiento

| Métrica | Valor |
|---------|-------|
| Memoria RAM | ~150MB (bot) + ~2GB (modelo embeddings) |
| Indexación PDF digital | ~1-2s por documento |
| Indexación PDF escaneado | ~5-10s por documento |
| Búsqueda | <100ms para 100K chunks |
| Compresión índice | 8x vs float32 |

## 🔒 Seguridad

- API keys en variables de entorno, nunca en código
- No se guardan PDFs originals (solo MD convertido)
- Bot solo responde en el grupo autorizado
- Sin telemetría ni analytics

## 📝 Licencia

MIT

---

Créditos: Lucas Mateo Tabares Franco + Ing. Jhoan Sebastian Bustamante Montes
