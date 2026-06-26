#!/usr/bin/env python3
"""
conversation_memory.py — Memoria conversacional para el bot de Telegram.
Almacena historial de conversaciones por grupo usando RedisVL.

MEJORA 8: RedisVL (Memoria conversacional)
No bloqueante: si Redis no está disponible, el bot funciona sin memoria.

Uso:
  from conversation_memory import ConversationMemory
  mem = ConversationMemory()
  mem.store("grupo_123", "¿Qué es café?", "El café es...")
  history = mem.get_context("grupo_123", limit=5)
"""

import json
import logging
import os
import time
from pathlib import Path

logger = logging.getLogger(__name__)

# === Configuración ===
HAS_REDISVL = False
try:
    from redisvl.extensions.message_history import MessageHistory
    HAS_REDISVL = True
except ImportError:
    HAS_REDISVL = False

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
MEMORY_FILE = Path.home() / ".hermes" / "conversation_memory.json"
MAX_HISTORY_PER_SESSION = 20  # Máximo de mensajes a recordar por sesión
TTL_HOURS = 24  # Tiempo de vida del historial


class ConversationMemory:
    """
    Memoria conversacional que almacena historial de conversaciones.
    Usa RedisVL si está disponible, fallback a JSON local.

    Args:
        redis_url: URL de Redis (default: REDIS_URL env o redis://localhost:6379)
        session_prefix: Prefijo para las sesiones
    """

    def __init__(self, redis_url: str = None, session_prefix: str = "telegram_bot"):
        self.redis_url = redis_url or REDIS_URL
        self.session_prefix = session_prefix
        self._redis_history = None
        self._local_memory: dict[str, list[dict]] = {}
        self._last_cleanup = time.time()

        # Intentar inicializar RedisVL
        self._init_redis()

        # Cargar memoria local (fallback)
        self._load_local()

    def _init_redis(self):
        """Intenta inicializar conexión RedisVL."""
        if not HAS_REDISVL:
            logger.debug("RedisVL no instalado, usando memoria local (JSON)")
            return

        try:
            from redis import Redis as RedisClient
            client = RedisClient.from_url(self.redis_url)
            client.ping()
            client.close()
            self._redis_available = True
            logger.info("✅ RedisVL disponible para memoria conversacional")
        except Exception as e:
            self._redis_available = False
            logger.debug(f"Redis no disponible ({e}), usando memoria local")

    def _load_local(self):
        """Carga memoria local desde archivo JSON."""
        if MEMORY_FILE.exists():
            try:
                self._local_memory = json.loads(MEMORY_FILE.read_text(encoding="utf-8"))
                logger.debug(f"Memoria local cargada: {len(self._local_memory)} sesiones")
            except Exception as e:
                logger.warning(f"Error cargando memoria local: {e}")
                self._local_memory = {}

    def _save_local(self):
        """Guarda memoria local a archivo JSON."""
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            MEMORY_FILE.write_text(
                json.dumps(self._local_memory, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning(f"Error guardando memoria local: {e}")

    def _get_session_key(self, session_id: str) -> str:
        """Genera clave de sesión."""
        return f"{self.session_prefix}:{session_id}"

    def store(self, session_id: str, prompt: str, response: str, metadata: dict = None) -> bool:
        """
        Almacena una interacción en la memoria.

        Args:
            session_id: ID de la sesión/grupo
            prompt: Pregunta del usuario
            response: Respuesta del bot
            metadata: Metadatos adicionales (opcional)

        Returns:
            True si se almacenó correctamente
        """
        entry = {
            "prompt": prompt,
            "response": response,
            "timestamp": time.time(),
            "session_id": session_id,
        }
        if metadata:
            entry["metadata"] = metadata

        # Intentar RedisVL primero
        if self._redis_available and HAS_REDISVL:
            try:
                key = self._get_session_key(session_id)
                if self._redis_history is None:
                    self._redis_history = MessageHistory(
                        name=f"{self.session_prefix}_history",
                        redis_url=self.redis_url,
                    )
                self._redis_history.store(prompt, response, session_tag=session_id)
                return True
            except Exception as e:
                logger.debug(f"Redis store falló ({e}), usando fallback local")
                self._redis_available = False

        # Fallback: memoria local
        if session_id not in self._local_memory:
            self._local_memory[session_id] = []
        self._local_memory[session_id].append(entry)

        # Limitar tamaño
        if len(self._local_memory[session_id]) > MAX_HISTORY_PER_SESSION:
            self._local_memory[session_id] = self._local_memory[session_id][-MAX_HISTORY_PER_SESSION:]

        self._save_local()
        return True

    def get_context(self, session_id: str, limit: int = 5) -> str:
        """
        Recupera el contexto de conversación reciente.

        Args:
            session_id: ID de la sesión/grupo
            limit: Número máximo de intercambios a recuperar

        Returns:
            String con el historial formateado para el prompt
        """
        history = self._get_history(session_id, limit)
        if not history:
            return ""

        context_parts = []
        for entry in history:
            context_parts.append(f"Usuario: {entry.get('prompt', '')}")
            context_parts.append(f"Asistente: {entry.get('response', '')}")

        return "\n".join(context_parts)

    def _get_history(self, session_id: str, limit: int = 5) -> list[dict]:
        """Recupera historial de una sesión."""
        # Intentar Redis
        if self._redis_available and HAS_REDISVL:
            try:
                if self._redis_history is None:
                    self._redis_history = MessageHistory(
                        name=f"{self.session_prefix}_history",
                        redis_url=self.redis_url,
                    )
                recent = self._redis_history.get_recent(
                    session_tag=session_id,
                    limit=limit,
                )
                return [
                    {"prompt": m.get("role", ""), "response": m.get("content", "")}
                    for m in recent
                ]
            except Exception as e:
                logger.debug(f"Redis get falló ({e}), usando fallback local")
                self._redis_available = False

        # Fallback local
        entries = self._local_memory.get(session_id, [])
        return entries[-limit:] if entries else []

    def clear_session(self, session_id: str) -> bool:
        """
        Limpia el historial de una sesión.

        Args:
            session_id: ID de la sesión/grupo

        Returns:
            True si se limpió correctamente
        """
        if self._redis_available and HAS_REDISVL:
            try:
                if self._redis_history is None:
                    self._redis_history = MessageHistory(
                        name=f"{self.session_prefix}_history",
                        redis_url=self.redis_url,
                    )
                self._redis_history.clear()
                return True
            except Exception as e:
                logger.debug(f"Redis clear falló ({e})")

        if session_id in self._local_memory:
            del self._local_memory[session_id]
            self._save_local()
        return True

    def stats(self) -> dict:
        """Estadísticas de la memoria."""
        total_sessions = 0
        total_entries = 0
        if self._redis_available:
            total_sessions = 0  # RedisVL no expone conteo fácilmente
        else:
            total_sessions = len(self._local_memory)
            total_entries = sum(len(v) for v in self._local_memory.values())

        return {
            "backend": "redis" if self._redis_available and HAS_REDISVL else "local_json",
            "redis_available": self._redis_available and HAS_REDISVL,
            "total_sessions": total_sessions,
            "total_entries": total_entries,
        }


# Singleton global
_memory_instance = None


def get_memory(redis_url: str = None) -> ConversationMemory:
    """Retorna la instancia singleton de ConversationMemory."""
    global _memory_instance
    if _memory_instance is None:
        _memory_instance = ConversationMemory(redis_url=redis_url)
    return _memory_instance


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Memoria conversacional")
    parser.add_argument("--test", action="store_true", help="Ejecutar prueba")
    parser.add_argument("--stats", action="store_true", help="Mostrar estadísticas")
    parser.add_argument("--clear", type=str, help="Limpiar sesión")
    args = parser.parse_args()

    mem = get_memory()

    if args.stats:
        print(json.dumps(mem.stats(), indent=2, ensure_ascii=False))

    if args.clear:
        mem.clear_session(args.clear)
        print(f"✅ Sesión '{args.clear}' limpiada")

    if args.test:
        print("🧪 Probando ConversationMemory...")
        mem.store("test_session", "Hola", "¡Hola! ¿En qué puedo ayudarte?")
        mem.store("test_session", "¿Qué es café?", "El café es una bebida...")
        ctx = mem.get_context("test_session", limit=5)
        print(f"Contexto recuperado ({len(ctx)} chars):")
        print(ctx[:300])
        print(f"\nBackend: {'RedisVL' if mem._redis_available else 'Local JSON'}")
