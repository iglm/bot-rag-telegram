#!/usr/bin/env python3
"""
bot_documentos_indexados.py — Bot RAG para grupo Telegram.
Indexa PDFs/MDs/JSONs/TXTs subidos al grupo y responde preguntas.

Mejoras aplicadas:
- Texto completo en metadata (no preview)
- OCR automático para PDFs (pymupdf4llm + ocrmypdf)
- asyncio.to_thread para no bloquear event loop
- Filtrado por GROUP_ID
- aiohttp.ClientSession reutilizable
- Rate limiting OpenRouter
- Limpieza de temporales al inicio
- Logging de queries
- Deduplicación por hash
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import Message

# Agregar scripts/ al path
sys.path.insert(0, str(Path.home() / "scripts"))
from turbovec_rag import RagEngine
from ocr_converter import convert_pdf_to_md

# --- MEJORA 2: LLMRouter (Routing inteligente de modelos) ---
HAS_LLMROUTER = False
try:
    from llm_router import select_model, classify_query
    HAS_LLMROUTER = True
except ImportError:
    HAS_LLMROUTER = False

# --- MEJORA 6: TruLens (Tracking de experimentos) ---
HAS_TRULENS = False
try:
    from trulens.core import TruSession
    from trulens.core.otel.instrument import instrument
    HAS_TRULENS = True
except ImportError:
    HAS_TRULENS = False

# --- MEJORA 8: Memoria conversacional ---
HAS_CONVERSATION_MEMORY = False
try:
    from conversation_memory import get_memory as get_conversation_memory
    HAS_CONVERSATION_MEMORY = True
except ImportError:
    HAS_CONVERSATION_MEMORY = False

# Archivo de logging para TruLens (si no usa base de datos)
TRULENS_LOG_FILE = Path.home() / ".hermes" / "trulens_queries.jsonl"

# Config — cargar .env si existe
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().strip().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

BOT_TOKEN = os.getenv("BOT_TOKEN_DOCS", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
GROUP_ID = -1003795352933
DOWNLOAD_DIR = Path("/tmp/bot_docs")
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_CONCURRENT_OPENROUTER = 3  # Rate limit: max 3 requests simultáneos

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Inicializar
bot: Bot = Bot(token=BOT_TOKEN) if BOT_TOKEN else None  # type: ignore
dp = Dispatcher()
engine: RagEngine | None = None

# Semáforo para rate limiting de OpenRouter
_openrouter_semaphore = asyncio.Semaphore(MAX_CONCURRENT_OPENROUTER)

# Session reutilizable de aiohttp
_http_session: aiohttp.ClientSession | None = None

# Cache de hashes para deduplicación
_indexed_hashes: set = set()


def get_engine() -> RagEngine:
    """Carga el engine una sola vez."""
    global engine
    if engine is None:
        engine = RagEngine()
        # Cargar hashes de documentos ya indexados
        for meta in engine.metadata:
            if "hash" in meta:
                _indexed_hashes.add(meta["hash"])
    return engine


def get_http_session() -> aiohttp.ClientSession:
    """Retorna la session reutilizable."""
    global _http_session
    if _http_session is None or _http_session.closed:
        _http_session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=10, ttl_dns_cache=300),
            timeout=aiohttp.ClientTimeout(total=60),
        )
    return _http_session


def file_hash(path: str) -> str:
    """Calcula SHA256 de un archivo."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def cleanup_temp():
    """Limpia temporales huérfanos al inicio."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for f in DOWNLOAD_DIR.iterdir():
        try:
            f.unlink()
            logger.info(f"🧹 Temporal limpiado: {f.name}")
        except Exception:
            pass


async def download_file(file_path: str, dest: Path) -> bool:
    """Descarga un archivo de Telegram."""
    try:
        if not bot:
            return False
        await bot.download_file(file_path, dest)  # type: ignore
        return True
    except Exception as e:
        logger.error(f"Error descargando {file_path}: {e}")
        return False


async def call_openrouter(prompt: str, context: str = "", session_id: str = None) -> str:
    """Llama a OpenRouter para generar respuesta con rate limiting.
    
    Args:
        prompt: Pregunta del usuario
        context: Contexto de documentos recuperados
        session_id: ID de sesión para memoria conversacional (opcional)
    """
    if not OPENROUTER_API_KEY:
        return "❌ OPENROUTER_API_KEY no configurada"

    # --- MEJORA 2: LLMRouter - elegir modelo según tipo de query ---
    model_to_use = OPENROUTER_MODEL
    if HAS_LLMROUTER:
        try:
            model_to_use = select_model(prompt)
            logger.debug(f"LLMRouter seleccionó: {model_to_use}")
        except Exception as e:
            logger.warning(f"LLMRouter falló ({e}), usando modelo por defecto")

    # --- MEJORA 8: Memoria conversacional ---
    conversation_context = ""
    if HAS_CONVERSATION_MEMORY and session_id:
        try:
            mem = get_conversation_memory()
            conversation_context = mem.get_context(session_id, limit=3)
            logger.debug(f"Contexto conversacional recuperado: {len(conversation_context)} chars")
        except Exception as e:
            logger.debug(f"Memoria conversacional no disponible: {e}")

    system_prompt = """Eres un asistente útil que responde preguntas basándose SOLO en los documentos proporcionados.
Si la información no está en los documentos, di que no sabes.
Responde en español, de forma concisa y directa.
Cita el nombre del documento cuando sea relevante."""

    if context:
        user_content = f"Documentos relevantes:\n{context}\n\nPregunta: {prompt}"
    else:
        user_content = prompt

    # Si hay contexto conversacional, agregarlo
    if conversation_context:
        user_content = f"Historial de conversación:\n{conversation_context}\n\n{user_content}"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model_to_use,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 1024,
        "temperature": 0.7,
    }

    async with _openrouter_semaphore:
        try:
            session = get_http_session()
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
            ) as resp:
                if resp.status == 429:
                    logger.warning("⚠️ Rate limit de OpenRouter, esperando 2s...")
                    await asyncio.sleep(2)
                    return await call_openrouter(prompt, context, session_id)

                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"OpenRouter error {resp.status}: {text[:200]}")
                    return f"❌ Error del modelo ({resp.status})"

                data = await resp.json()
                response_text = data["choices"][0]["message"]["content"]

                # --- MEJORA 6: TruLens logging ---
                _log_query_to_trulens({
                    "prompt": prompt,
                    "context": context[:500] if context else "",
                    "model": model_to_use,
                    "response": response_text[:500],
                    "session_id": session_id,
                    "timestamp": time.time(),
                })

                # --- MEJORA 8: Guardar en memoria conversacional ---
                if HAS_CONVERSATION_MEMORY and session_id:
                    try:
                        mem = get_conversation_memory()
                        mem.store(session_id, prompt, response_text)
                    except Exception as e:
                        logger.debug(f"Error guardando memoria: {e}")

                return response_text

        except asyncio.TimeoutError:
            return "❌ Timeout del modelo"
        except Exception as e:
            logger.error(f"Error OpenRouter: {e}")
            return f"❌ Error de conexión: {e}"


def _log_query_to_trulens(data: dict):
    """Guarda datos de query/respuesta para tracking con TruLens.
    No bloqueante: si falla, el bot sigue funcionando.
    """
    if not HAS_TRULENS:
        return
    try:
        TRULENS_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TRULENS_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.debug(f"TruLens logging falló: {e}")


@dp.message(Command("start"))
async def cmd_start(message: Message):
    """Mensaje de bienvenida."""
    if message.chat.id != GROUP_ID:
        return
    await message.reply(
        "📚 **Bot de Documentos Indexados**\n\n"
        "Soy un asistente que busca en los documentos subidos a este grupo.\n\n"
        "**Comandos:**\n"
        "• `/indexar` — Indexar todos los documentos del chat\n"
        "• `/estado` — Ver estado del índice\n"
        "• `/buscar <pregunta>` — Buscar en documentos\n"
        "• `/memoria` — Estado de la memoria conversacional\n"
        "• `/clear_memoria` — Limpiar memoria conversacional\n"
        "• `/help` — Ayuda\n\n"
        "También puedes hacer preguntas directamente y buscaré en los documentos.",
        parse_mode="Markdown",
    )


@dp.message(Command("estado"))
async def cmd_estado(message: Message):
    """Muestra estado del índice."""
    if message.chat.id != GROUP_ID:
        return
    eng = get_engine()
    if not eng.metadata:
        await message.reply("📭 No hay documentos indexados. Usa `/indexar` primero.")
        return

    pdfs = set(m["pdf"] for m in eng.metadata)
    total = len(eng.metadata)
    await message.reply(
        f"📊 **Estado del índice:**\n"
        f"• Documentos: {len(pdfs)}\n"
        f"• Chunks indexados: {total}\n"
        f"• Compresión: 8x (turbovec 4-bit)\n"
        f"• Dimensión: {eng.dim}",
        parse_mode="Markdown",
    )


@dp.message(Command("indexar"))
async def cmd_indexar(message: Message):
    """Indexa documentos del chat."""
    if message.chat.id != GROUP_ID:
        return

    await message.reply("⏳ Indexando documentos... Esto puede tardar unos minutos.")

    eng = get_engine()

    try:
        messages = []
        if not bot:
            await message.reply("❌ Bot no inicializado.")
            return

        async for msg in bot.get_chat_history(message.chat.id, limit=200):  # type: ignore
            if msg.document:
                messages.append(msg)

        if not messages:
            await message.reply("📭 No encontré documentos en el historial reciente.")
            return

        indexed = 0
        skipped = 0
        for msg in messages:
            doc = msg.document
            if not doc:
                continue

            file_name = doc.file_name or "unknown"
            if not file_name.lower().endswith((".pdf", ".md", ".json", ".txt")):
                continue

            if (doc.file_size or 0) > MAX_FILE_SIZE:
                continue

            # Descargar
            dest = DOWNLOAD_DIR / file_name
            DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

            file = await bot.get_file(doc.file_id) if bot else None  # type: ignore
            if not file:
                continue

            success = await download_file(file.file_path, dest)
            if not success:
                continue

            # Deduplicación por hash (QA-2)
            fhash = file_hash(str(dest))
            if fhash in _indexed_hashes:
                logger.info(f"⏭️  Duplicado, saltando: {file_name}")
                skipped += 1
                dest.unlink(missing_ok=True)
                continue
            _indexed_hashes.add(fhash)

            try:
                if file_name.lower().endswith(".pdf"):
                    # PDF → convertir a MD con OCR
                    md_path = str(dest.with_suffix(".md"))
                    await asyncio.get_event_loop().run_in_executor(
                        None, convert_pdf_to_md, str(dest), md_path
                    )
                    count = await asyncio.get_event_loop().run_in_executor(
                        None, eng.index_file, md_path
                    )
                    # Guardar hash en metadata
                    if eng.metadata:
                        eng.metadata[-1]["hash"] = fhash
                    indexed += 1
                    logger.info(f"Indexado (OCR): {file_name} ({count} chunks)")
                    Path(md_path).unlink(missing_ok=True)
                else:
                    count = await asyncio.get_event_loop().run_in_executor(
                        None, eng.index_file, str(dest)
                    )
                    if eng.metadata:
                        eng.metadata[-1]["hash"] = fhash
                    indexed += 1
                    logger.info(f"Indexado: {file_name} ({count} chunks)")

            except Exception as e:
                logger.error(f"Error indexando {file_name}: {e}")
            finally:
                if dest.exists():
                    dest.unlink()

        if indexed > 0:
            await asyncio.get_event_loop().run_in_executor(None, eng.save)
            await message.reply(
                f"✅ **Indexación completa:**\n"
                f"• Documentos indexados: {indexed}\n"
                f"• Saltados (duplicados): {skipped}\n"
                f"• Total chunks: {len(eng.metadata)}",
                parse_mode="Markdown",
            )
        else:
            await message.reply(f"⚠️ No se indexaron documentos nuevos. Saltados: {skipped}")

    except Exception as e:
        logger.error(f"Error en indexación: {e}")
        await message.reply("❌ Error durante la indexación.")


@dp.message(Command("buscar"))
async def cmd_buscar(message: Message):
    """Busca en los documentos."""
    if message.chat.id != GROUP_ID:
        return

    query = (message.text or "").replace("/buscar", "").strip()
    if not query:
        await message.reply("Uso: `/buscar <pregunta>`", parse_mode="Markdown")
        return

    logger.info(f"QUERY: {query}")

    await message.reply("🔍 Buscando...")

    eng = get_engine()
    results = await asyncio.get_event_loop().run_in_executor(None, eng.query, query, 5)

    if not results:
        await message.reply("📭 No encontré información relevante en los documentos.")
        return

    # Construir contexto completo (BUG #1 fix: usar texto completo, no preview)
    context_parts = []
    for r in results:
        full_text = r.get("text", r.get("text_preview", ""))
        context_parts.append(f"[{r['pdf']}]: {full_text[:500]}")

    context = "\n\n".join(context_parts)

    respuesta = await call_openrouter(query, context, session_id=str(message.chat.id))

    await message.reply(respuesta, parse_mode="Markdown")


@dp.message(Command("help"))
async def cmd_help(message: Message):
    """Ayuda."""
    if message.chat.id != GROUP_ID:
        return
    await message.reply(
        "📚 **Comandos disponibles:**\n\n"
        "• `/indexar` — Busca y indexa documentos del chat\n"
        "• `/estado` — Estado del índice\n"
        "• `/buscar <pregunta>` — Búsqueda con respuesta IA\n"
        "• `/memoria` — Estado de la memoria conversacional\n"
        "• `/clear_memoria` — Limpiar memoria conversacional\n"
        "• `/help` — Esta ayuda\n\n"
        "💡 También puedes hacer preguntas directamente.",
        parse_mode="Markdown",
    )


@dp.message(Command("memoria"))
async def cmd_memoria(message: Message):
    """Muestra estado de la memoria conversacional."""
    if message.chat.id != GROUP_ID:
        return
    if not HAS_CONVERSATION_MEMORY:
        await message.reply("💾 Memoria conversacional no disponible (RedisVL no instalado)")
        return
    try:
        mem = get_conversation_memory()
        stats = mem.stats()
        await message.reply(
            f"💾 **Memoria conversacional:**\n"
            f"• Backend: {stats.get('backend', 'N/A')}\n"
            f"• Redis disponible: {stats.get('redis_available', False)}\n"
            f"• Sesiones: {stats.get('total_sessions', 0)}\n"
            f"• Entradas: {stats.get('total_entries', 0)}",
            parse_mode="Markdown",
        )
    except Exception as e:
        await message.reply(f"❌ Error: {e}")


@dp.message(Command("clear_memoria"))
async def cmd_clear_memoria(message: Message):
    """Limpia la memoria conversacional de este grupo."""
    if message.chat.id != GROUP_ID:
        return
    if not HAS_CONVERSATION_MEMORY:
        await message.reply("💾 Memoria conversacional no disponible")
        return
    try:
        mem = get_conversation_memory()
        mem.clear_session(str(message.chat.id))
        await message.reply("✅ Memoria conversacional limpiada")
    except Exception as e:
        await message.reply(f"❌ Error: {e}")


@dp.message(F.document)
async def handle_document(message: Message):
    """Cuando suben un archivo, convertir a MD e indexar."""
    if message.chat.id != GROUP_ID:
        return

    doc = message.document
    if not doc:
        return

    file_name = doc.file_name or "unknown"
    if not file_name.lower().endswith((".pdf", ".md", ".json", ".txt")):
        return

    if not bot:
        return

    if (doc.file_size or 0) > MAX_FILE_SIZE:
        await message.reply(f"⚠️ Archivo muy grande ({(doc.file_size or 0) // 1024 // 1024}MB). Máximo: 50MB")
        return

    await message.reply(f"📄 Procesando: {file_name}...")

    eng = get_engine()
    dest = DOWNLOAD_DIR / file_name
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    file = await bot.get_file(doc.file_id) if bot else None  # type: ignore
    if not file:
        await message.reply("❌ No pude acceder al archivo.")
        return

    success = await download_file(file.file_path, dest)
    if not success:
        await message.reply("❌ Error descargando el archivo.")
        return

    try:
        # Deduplicación
        fhash = file_hash(str(dest))
        if fhash in _indexed_hashes:
            await message.reply(f"⏭️  Archivo duplicado, ya estaba indexado: {file_name}")
            dest.unlink(missing_ok=True)
            return
        _indexed_hashes.add(fhash)

        if file_name.lower().endswith(".pdf"):
            # PDF → MD con OCR
            md_path = str(dest.with_suffix(".md"))
            await message.reply("🤖 Analizando y convirtiendo PDF a Markdown...")

            await asyncio.get_event_loop().run_in_executor(
                None, convert_pdf_to_md, str(dest), md_path
            )

            count = await asyncio.get_event_loop().run_in_executor(
                None, eng.index_file, md_path
            )
            if eng.metadata:
                eng.metadata[-1]["hash"] = fhash

            await asyncio.get_event_loop().run_in_executor(None, eng.save)

            dest.unlink(missing_ok=True)
            Path(md_path).unlink(missing_ok=True)

            await message.reply(
                f"✅ **{file_name}** convertido e indexado:\n"
                f"• Formato: PDF → Markdown (OCR)\n"
                f"• Chunks: {count}",
                parse_mode="Markdown",
            )
        else:
            # MD/JSON/TXT → indexar directo
            count = await asyncio.get_event_loop().run_in_executor(
                None, eng.index_file, str(dest)
            )
            if eng.metadata:
                eng.metadata[-1]["hash"] = fhash

            await asyncio.get_event_loop().run_in_executor(None, eng.save)

            await message.reply(f"✅ Indexado: {file_name} ({count} chunks)")

    except Exception as e:
        logger.error(f"Error procesando: {e}")
        await message.reply("❌ Error al procesar el archivo.")
    finally:
        if dest.exists():
            dest.unlink()
        md_temp = dest.with_suffix(".md")
        if md_temp.exists():
            md_temp.unlink()


@dp.message(F.text)
async def handle_question(message: Message):
    """Responde preguntas de texto buscando en documentos."""
    if message.chat.id != GROUP_ID:
        return

    text = message.text or ""
    query = text.strip()
    if not query or len(query) < 3:
        return

    if query.startswith("/"):
        return

    eng = get_engine()
    if not eng or not eng.metadata:
        return

    logger.info(f"QUERY: {query}")

    await message.reply("🔍 Buscando...")

    results = await asyncio.get_event_loop().run_in_executor(None, eng.query, query, 5)

    if not results:
        await message.reply("📭 No encontré información relevante en los documentos.")
        return

    # Contexto completo (BUG #1 fix)
    context_parts = []
    for r in results:
        full_text = r.get("text", r.get("text_preview", ""))
        context_parts.append(f"[{r['pdf']}]: {full_text[:500]}")

    context = "\n\n".join(context_parts)

    respuesta = await call_openrouter(query, context, session_id=str(message.chat.id))

    await message.reply(respuesta, parse_mode="Markdown")


async def on_shutdown():
    """Limpieza al cerrar."""
    global _http_session
    if _http_session and not _http_session.closed:
        await _http_session.close()
    cleanup_temp()


async def main():
    """Inicia el bot."""
    global engine

    if not BOT_TOKEN:
        print("❌ BOT_TOKEN_DOCS no configurada")
        sys.exit(1)

    if not OPENROUTER_API_KEY:
        print("⚠️ OPENROUTER_API_KEY no configurada")

    # OPT: uvloop para mejor async throughput
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
        print("✅ uvloop activado")
    except ImportError:
        print("⚠️ uvloop no disponible (pip install uvloop)")

    # OPT: TORCH_NUM_THREADS
    os.environ.setdefault("TORCH_NUM_THREADS", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "2")

    # Limpieza de temporales
    cleanup_temp()

    # Cargar engine
    engine = get_engine()

    print(f"🤖 Bot iniciado en grupo {GROUP_ID}")
    print(f"📊 Engine: turbovec {engine.dim}-dim 4-bit")
    print(f"🧠 Modelo: {OPENROUTER_MODEL}")

    dp.shutdown.register(on_shutdown)
    await dp.start_polling(bot)  # type: ignore


if __name__ == "__main__":
    asyncio.run(main())
