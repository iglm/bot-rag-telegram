# 🤖 Telegram RAG Bot — NotebookLM-like con TurboQuant

Bot Telegram para indexar documentos y responder preguntas usando RAG (Retrieval-Augmented Generation) 100% local y gratis.

## 🎯 Características

- **Indexación automática** de PDFs, MDs, JSONs y TXTs en grupos Telegram
- **Conversión OCR inteligente**: detecta tipo de PDF y elige la mejor herramienta
- **Embeddings profesionales** con BAAI/bge-m3 (multilingüe, 1024-dim)
- **Compresión 8x** con turbovec 4-bit (Google TurboQuant)
- **Respuestas IA** via OpenRouter (modelos gratuitos)
- **Deduplicación** por hash SHA256
- **Rate limiting** para OpenRouter
- **$0/mes** — todo corre en VPS sin GPU

## 📊 Arquitectura

```
📄 Documento subido (PDF/MD/JSON/TXT)
       ↓
🤖 Agente clasificador OCR (analiza contenido)
       ↓
┌─────────────────────────────────────────┐
│ PDF digital    → pymupdf4llm (rápido)   │
│ Escaneado      → OCRmyPDF (Tesseract)   │
│ Largo/complejo → Unlimited-OCR (Baidu)  │
│ MD/JSON/TXT    → indexación directa      │
└─────────────────────────────────────────┘
       ↓
🗄️ TurboQuant Index (turbovec 4-bit, 1024-dim)
       ↓
🔍 Retrieval top-k → contexto completo
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
- Tesseract OCR instalado (`apt install tesseract-ocr tesseract-ocr-spa`)

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
# Editar .env con tus credenciales:
# BOT_TOKEN_DOCS=tu_token_de_botfather
# OPENROUTER_API_KEY=tu_key_de_openrouter

# 5. Iniciar
python3 bot_documentos_indexados.py
```

### Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto:

```env
BOT_TOKEN_DOCS=8867204784:AAE...
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=nvidia/nemotron-3-super-120b-a12b:free
```

| Variable | Requerido | Descripción | Cómo obtener |
|----------|-----------|-------------|--------------|
| `BOT_TOKEN_DOCS` | ✅ | Token del bot de Telegram | Crear bot con @BotFather en Telegram |
| `OPENROUTER_API_KEY` | ✅ | API key de OpenRouter | Registrarse en https://openrouter.ai/keys |
| `OPENROUTER_MODEL` | Opcional | Modelo LLM para respuestas | Ver modelos gratuitos abajo |

### Modelos OpenRouter Gratuitos

| Modelo | Calidad | Contexto | Ideal para |
|--------|---------|----------|------------|
| `nvidia/nemotron-3-super-120b-a12b:free` | ⭐⭐⭐⭐⭐ | 1M | Mejor calidad (default) |
| `qwen/qwen3-coder:free` | ⭐⭐⭐⭐ | 1M | Respuestas generales |
| `google/gemma-4-31b-it:free` | ⭐⭐⭐⭐ | 256K | Multilingüe |
| `google/lyria-3-pro-preview` | ⭐⭐⭐⭐⭐ | 1M | Google quality |

## 📋 Uso

### En el grupo Telegram

1. **Crea un grupo** en Telegram
2. **Agrega el bot** como administrador
3. **Sube documentos** (PDF, MD, JSON, TXT) — el bot indexa automáticamente
4. **Haz preguntas** directamente en el grupo

| Comando | Acción |
|---------|--------|
| Subir archivo | Indexación automática |
| `/indexar` | Re-indexar todos los documentos del chat |
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

## 🔧 Configuración OCR

El agente clasificador analiza cada PDF y decide automáticamente:

| Tipo de PDF | Herramienta | Cuando |
|-------------|-------------|--------|
| Digital (texto seleccionable) | pymupdf4llm | >30% páginas con texto |
| Escaneado simple | OCRmyPDF | <100 páginas, >50% sin texto |
| Largo/complejo | Unlimited-OCR | >50 páginas escaneadas |
| Mixto | pymupdf4llm | Texto + imágenes |

## 📁 Estructura del Proyecto

```
bot-rag-telegram/
├── bot_documentos_indexados.py   # Bot Telegram (aiogram)
├── turbovec_rag.py              # Motor RAG (embeddings + turbovec)
├── ocr_classifier.py            # Agente clasificador OCR
├── ocr_converter.py             # Conversor PDF → MD (3 herramientas)
├── requirements.txt             # Dependencias Python
├── install.sh                   # Script de instalación automatizada
├── bot-docs-indexados.service   # Servicio systemd (producción)
├── .env.example                 # Plantilla de variables
├── .gitignore                   # Excluye .env, venv, modelos
└── README.md                    # Este archivo
```

## 🔒 Seguridad

- **API keys en `.env`** (no en código ni systemd)
- **`.env` con permisos 600** (solo el dueño puede leerlo)
- **`.gitignore`** excluye `.env`, modelos, venv
- **Bot solo responde en el grupo autorizado** (GROUP_ID filter)
- **Sin telemetría** ni analytics
- **No se guardan PDFs originales** (solo MD convertido)

## 📊 Rendimiento

| Métrica | Valor |
|---------|-------|
| Memoria RAM | ~150MB (bot) + ~2GB (modelo embeddings) |
| Indexación PDF digital | ~1-2s por documento |
| Indexación PDF escaneado | ~5-10s por documento |
| Búsqueda | <100ms para 100K chunks |
| Compresión índice | 8x vs float32 |
| Disco para 100 PDFs | ~30MB (MD) vs ~3GB (PDFs) |

## 🔄 Mantenimiento

### Comandos systemd

```bash
sudo systemctl start bot-docs-indexados    # Iniciar
sudo systemctl stop bot-docs-indexados     # Detener
sudo systemctl restart bot-docs-indexados  # Reiniciar
sudo systemctl status bot-docs-indexados   # Ver estado
journalctl -u bot-docs-indexados -f       # Ver logs en vivo
```

### Sincronización automática

El proyecto incluye un cronjob semanal que sincroniza automáticamente los cambios a GitHub:

```bash
# Se ejecuta cada domingo a las 06:00 UTC
# Sincroniza scripts/ → repo GitHub
```

### Re-indexación

Si cambias los documentos o quieres reconstruir el índice:

1. Borra `~/.hermes/turbovec_index/`
2. Ejecuta `/indexar` en el grupo Telegram

## 🐛 Troubleshooting

| Problema | Solución |
|----------|----------|
| Bot no responde | `sudo systemctl status bot-docs-indexados` |
| Error "ModuleNotFoundError" | `pip install -r requirements.txt` |
| OCR no funciona | `apt install tesseract-ocr tesseract-ocr-spa` |
| OpenRouter rate limit | El bot maneja rate limiting automáticamente (reintentos) |
| Disco lleno | El bot no guarda PDFs, solo MD liviano |
| Memoria insuficiente | El modelo bge-m3 usa ~1.4GB; cerrar otros bots si es necesario |

## 📝 Notas Importantes

- El **modelo de embeddings** (bge-m3) se descarga la primera vez (~1.4GB)
- Los **archivos temporales** se limpian automáticamente al iniciar
- La **deduplicación** evita re-indexar el mismo archivo
- El **contexto completo** de cada chunk se almacena en metadata (no solo preview)
- **No necesitas GPU** — todo corre en CPU

## 📜 Licencia

MIT

---

**Repositorio mantenido por:** Lucas Mateo Tabares Franco
**Créditos:** Lucas Mateo Tabares Franco + Ing. Jhoan Sebastian Bustamante Montes
