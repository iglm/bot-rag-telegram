#!/usr/bin/env python3
"""
bot_documentos_indexados.py — Bot RAG para grupo Telegram.
Indexa PDFs subidos al grupo y responde preguntas usando turbovec + OpenRouter.

Uso:
  python3 bot_documentos_indexados.py          # iniciar bot
  python3 bot_documentos_indexados.py --index  # solo indexar (modo batch)
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import aiohttp
import numpy as np
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.types import Message, ContentType

# Agregar scripts/ al path
sys.path.insert(0, str(Path.home() / "scripts"))
from turbovec_rag import RagEngine
from ocr_converter import convert_pdf_to_md

# Config
BOT_TOKEN = os.getenv("BOT_TOKEN_DOCS", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
GROUP_ID = -1003795352933
DOWNLOAD_DIR = Path("/tmp/bot_docs")
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Inicializar
bot: Bot = Bot(token=BOT_TOKEN) if BOT_TOKEN else None  # type: ignore
dp = Dispatcher()
engine: RagEngine | None = None  # Se carga lazy


def get_engine() -> RagEngine:
    """Carga el engine una sola vez."""
    global engine
    if engine is None:
        engine = RagEngine()
    return engine


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


async def call_openrouter(prompt: str, context: str = "") -> str:
    """Llama a OpenRouter para generar respuesta."""
    if not OPENROUTER_API_KEY:
        return "❌ OPENROUTER_API_KEY no configurada"

    system_prompt = """Eres un asistente útil que responde preguntas basándose SOLO en los documentos proporcionados.
Si la información no está en los documentos, di que no sabes.
Responde en español, de forma concisa y directa.
Cita el nombre del documento cuando sea relevante."""

    if context:
        user_content = f"Documentos relevantes:\n{context}\n\nPregunta: {prompt}"
    else:
        user_content = prompt

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 1024,
        "temperature": 0.7,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                logger.error(f"OpenRouter error {resp.status}: {text[:200]}")
                return f"❌ Error del modelo ({resp.status})"

            data = await resp.json()
            return data["choices"][0]["message"]["content"]


@dp.message(Command("start"))
async def cmd_start(message: Message):
    """Mensaje de bienvenida."""
    await message.reply(
        "📚 **Bot de Documentos Indexados**\n\n"
        "Soy un asistente que busca en los PDFs subidos a este grupo.\n\n"
        "**Comandos:**\n"
        "• `/indexar` — Indexar todos los PDFs del chat\n"
        "• `/estado` — Ver estado del índice\n"
        "• `/buscar <pregunta>` — Buscar en documentos\n\n"
        "También puedes hacer preguntas directamente y buscaré en los documentos.",
        parse_mode="Markdown",
    )


@dp.message(Command("estado"))
async def cmd_estado(message: Message):
    """Muestra estado del índice."""
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
    """Indexa PDFs del chat."""
    await message.reply("⏳ Indexando documentos... Esto puede tardar unos minutos.")

    eng = get_engine()

    # Buscar mensajes con documentos recientes
    # Nota: aiogram no tiene historial directo, indexamos los últimos 100 mensajes
    if not bot:
        await message.reply("❌ Bot no inicializado.")
        return

    try:
        messages = []
        async for msg in bot.get_chat_history(message.chat.id, limit=200):  # type: ignore
            if msg.document:
                messages.append(msg)

        if not messages:
            await message.reply("📭 No encontré documentos en el historial reciente.")
            return

        indexed = 0
        for msg in messages:
            doc = msg.document
            if not doc or not doc.file_name.lower().endswith(".pdf"):
                continue

            if doc.file_size > MAX_FILE_SIZE:
                logger.info(f"Archivo muy grande, saltando: {doc.file_name}")
                continue

            # Descargar
            dest = DOWNLOAD_DIR / doc.file_name
            DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

            file = await bot.get_file(doc.file_id) if bot else None
            if not file:
                continue

            success = await download_file(file.file_path, dest)
            if not success:
                continue

            # Indexar
            try:
                if doc.file_name and doc.file_name.lower().endswith(".pdf"):
                    # PDF → convertir a MD
                    md_path = str(dest.with_suffix(".md"))
                    await asyncio.get_event_loop().run_in_executor(
                        None, convert_pdf_to_md, str(dest), md_path
                    )
                    count = eng.index_file(md_path)
                    indexed += 1
                    logger.info(f"Indexado (OCR): {doc.file_name} ({count} chunks)")
                    # Limpiar MD temporal
                    Path(md_path).unlink(missing_ok=True)
                else:
                    count = eng.index_file(str(dest))
                    indexed += 1
                    logger.info(f"Indexado: {doc.file_name} ({count} chunks)")
            except Exception as e:
                logger.error(f"Error indexando {doc.file_name}: {e}")
            finally:
                # Limpiar temporal
                if dest.exists():
                    dest.unlink()

        if indexed > 0:
            eng.save()
            await message.reply(
                f"✅ **Indexación completa:**\n"
                f"• PDFs indexados: {indexed}\n"
                f"• Total chunks: {len(eng.metadata)}",
                parse_mode="Markdown",
            )
        else:
            await message.reply("⚠️ No se pudieron indexar documentos.")

    except Exception as e:
        logger.error(f"Error en indexación: {e}")
        await message.reply(f"❌ Error durante la indexación: {e}")


@dp.message(Command("buscar"))
async def cmd_buscar(message: Message):
    """Busca en los documentos."""
    query = (message.text or "").replace("/buscar", "").strip()
    if not query:
        await message.reply("Uso: `/buscar <pregunta>`", parse_mode="Markdown")
        return

    await message.reply("🔍 Buscando...")

    eng = get_engine()
    results = eng.query(query, top_k=5)

    if not results:
        await message.reply("📭 No encontré resultados. Indexa primero con `/indexar`.")
        return

    # Construir contexto
    context_parts = []
    for r in results:
        context_parts.append(f"[{r['pdf']}]: {r['text_preview'][:300]}")

    context = "\n\n".join(context_parts)

    # Llamar al LLM
    respuesta = await call_openrouter(query, context)

    await message.reply(
        f"🔍 **Resultados para:** {query}\n\n{respuesta}",
        parse_mode="Markdown",
    )


@dp.message(Command("help"))
async def cmd_help(message: Message):
    """Ayuda."""
    await message.reply(
        "📚 **Comandos disponibles:**\n\n"
        "• `/indexar` — Busca y indexa PDFs del chat\n"
        "• `/estado` — Estado del índice\n"
        "• `/buscar <pregunta>` — Búsqueda con respuesta IA\n"
        "• `/help` — Esta ayuda\n\n"
        "💡 También puedes hacer preguntas directamente y el bot buscará en los documentos.",
        parse_mode="Markdown",
    )


@dp.message(F.document)
async def handle_document(message: Message):
    """Cuando suben un archivo, convertir a MD e indexar."""
    doc = message.document
    if not doc or not (doc.file_name or "").lower().endswith((".pdf", ".md", ".json", ".txt")):
        return

    if not bot:
        return

    file_name = doc.file_name or "unknown"
    suffix = Path(file_name).suffix.lower()

    if (doc.file_size or 0) > MAX_FILE_SIZE:
        await message.reply(f"⚠️ Archivo muy grande ({(doc.file_size or 0) // 1024 // 1024}MB). Máximo: 50MB")
        return

    await message.reply(f"📄 Procesando: {file_name}...")

    # Si no es PDF, indexar directamente
    if suffix in (".md", ".json", ".txt"):
        eng = get_engine()
        dest = DOWNLOAD_DIR / file_name
        DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

        file = await bot.get_file(doc.file_id) if bot else None
        if not file:
            await message.reply("❌ No pude acceder al archivo.")
            return

        success = await download_file(file.file_path, dest)
        if not success:
            await message.reply("❌ Error descargando el archivo.")
            return

        try:
            count = eng.index_file(str(dest))
            eng.save()
            await message.reply(f"✅ Indexado: {file_name} ({count} chunks)")
        except Exception as e:
            logger.error(f"Error indexando: {e}")
            await message.reply(f"❌ Error indexando: {e}")
        finally:
            if dest.exists():
                dest.unlink()
        return

    # Es PDF → convertir a MD con OCR
    eng = get_engine()
    dest = DOWNLOAD_DIR / file_name
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    file = await bot.get_file(doc.file_id) if bot else None
    if not file:
        await message.reply("❌ No pude acceder al archivo.")
        return

    success = await download_file(file.file_path, dest)
    if not success:
        await message.reply("❌ Error descargando el PDF.")
        return

    try:
        # Convertir PDF a MD automáticamente
        md_path = str(dest.with_suffix(".md"))
        await message.reply("🤖 Analizando y convirtiendo PDF a Markdown...")

        result = await asyncio.get_event_loop().run_in_executor(
            None, convert_pdf_to_md, str(dest), md_path
        )

        # Indexar el MD resultante
        count = eng.index_file(md_path)
        eng.save()

        # Borrar PDF original, mantener MD
        dest.unlink()
        md_size = os.path.getsize(md_path) / 1024

        await message.reply(
            f"✅ **{file_name}** convertido e indexado:\n"
            f"• Formato: PDF → Markdown (OCR)\n"
            f"• Chunks: {count}\n"
            f"• MD generado: {md_size:.1f} KB",
            parse_mode="Markdown",
        )

    except Exception as e:
        logger.error(f"Error en conversión OCR: {e}")
        await message.reply(f"❌ Error: {e}")
    finally:
        # Limpiar temporales
        if dest.exists():
            dest.unlink()
        md_temp = dest.with_suffix(".md")
        if md_temp.exists():
            md_temp.unlink()


@dp.message(F.text)
async def handle_question(message: Message):
    """Responde preguntas de texto buscando en documentos."""
    text = message.text or ""
    query = text.strip()
    if not query or len(query) < 3:
        return

    # Ignorar comandos
    if query.startswith("/"):
        return

    # Solo responder si hay documentos indexados
    eng = get_engine()
    if not eng or not eng.metadata:
        return  # No responder si no hay nada indexado

    await message.reply("🔍 Buscando...")

    results = eng.query(query, top_k=5)

    if not results:
        await message.reply("📭 No encontré información relevante en los documentos.")
        return

    # Construcción contexto
    context_parts = []
    for r in results:
        context_parts.append(f"[{r['pdf']}]: {r['text_preview'][:300]}")

    context = "\n\n".join(context_parts)

    respuesta = await call_openrouter(query, context)

    await message.reply(respuesta, parse_mode="Markdown")


async def main():
    """Inicia el bot."""
    if not BOT_TOKEN:
        print("❌ BOT_TOKEN_DOCS no configurada. export BOT_TOKEN_DOCS=tu_token")
        sys.exit(1)

    if not OPENROUTER_API_KEY:
        print("⚠️ OPENROUTER_API_KEY no configurada. Las respuestas no tendrán LLM.")

    print(f"🤖 Bot iniciado en grupo {GROUP_ID}")
    if engine:
        print(f"📊 Engine: turbovec {engine.dim}-dim 4-bit")
    print(f"🧠 Modelo: {OPENROUTER_MODEL}")
    await dp.start_polling(bot)  # type: ignore


if __name__ == "__main__":
    asyncio.run(main())
