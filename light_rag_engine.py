#!/usr/bin/env python3
"""
light_rag_engine.py — RAG con grafos de conocimiento usando LightRAG.
Alternativa a turbovec_rag.py para preguntas complejas que requieren
navegación de relaciones entre entidades.

Basado en: lightrag-hku (https://github.com/hkuds/lightrag)

Características:
  - Extrae entidades y relaciones del texto
  - Construye un grafo de conocimiento
  - Navega el grafo para encontrar contexto relevante
  - Modos de búsqueda: local (entidades cercanas), global (conceptos amplios),
    hybrid (combinado), naive (vectorial)

Uso:
  python3 light_rag_engine.py index <documento> [--working-dir ./rag_storage]
  python3 light_rag_engine.py index-dir <directorio> [--working-dir ./rag_storage]
  python3 light_rag_engine.py query <pregunta> [--mode hybrid] [--top-k 40]

Dependencias:
  pip install lightrag-hku

Nota:
  - Requiere una LLM configurada (OPENAI_API_KEY o similar).
  - Usa sentence-transformers para embeddings por defecto.
  - Es una alternativa para preguntas complejas, no reemplaza RAG vectorial.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Verificar disponibilidad de LightRAG
try:
    from lightrag import LightRAG, QueryParam
    HAS_LIGHTRAG = True
except ImportError:
    HAS_LIGHTRAG = False
    logger.warning("⚠️  lightrag-hku no instalado. Ejecuta: pip install lightrag-hku")


class LightRagEngine:
    """
    Motor RAG basado en grafos de conocimiento usando LightRAG.
    Alternativa al RAG vectorial tradicional (turbovec_rag.py).

    Extrae entidades y relaciones del texto, construye un grafo,
    y navega relaciones para responder preguntas complejas.
    """

    def __init__(self, working_dir: str = None, llm_model_name: str = None,
                 embedding_model_name: str = "BAAI/bge-m3"):
        """
        Inicializa el motor LightRAG.

        Args:
            working_dir: Directorio para persistencia del grafo y vectores
            llm_model_name: Nombre del modelo LLM (ej: "gpt-4o-mini", "claude-3-haiku")
            embedding_model_name: Nombre del modelo de embeddings
        """
        if not HAS_LIGHTRAG:
            raise ImportError(
                "lightrag-hku no está instalado.\n"
                "  pip install lightrag-hku\n\n"
                "LightRAG con grafos requiere este paquete."
            )

        if working_dir is None:
            working_dir = str(Path.home() / ".hermes" / "lightrag_storage")

        self.working_dir = working_dir
        self._model_name = embedding_model_name
        self._llm_model_name = llm_model_name or os.getenv("LIGHTRAG_LLM_MODEL", "gpt-4o-mini")
        self.rag = None
        self._init_rag()

    def _init_rag(self):
        """Inicializa la instancia LightRAG."""
        Path(self.working_dir).mkdir(parents=True, exist_ok=True)

        # Configurar embeddings con sentence-transformers (bge-m3)
        from lightrag.llm import openai_complete_if_cache
        from lightrag.embed import hf_embed
        from sentence_transformers import SentenceTransformer

        logger.info(f"🔧 Inicializando LightRAG (working_dir={self.working_dir})")
        logger.info(f"   Modelo embeddings: {self._model_name}")
        logger.info(f"   Modelo LLM: {self._llm_model_name}")

        async def llm_model_func(prompt, system_prompt=None, history_messages=None,
                                  keyword_extraction=False, **kwargs) -> str:
            """Función LLM para LightRAG."""
            if self._llm_model_name.startswith("gpt") or self._llm_model_name.startswith("o"):
                return await openai_complete_if_cache(
                    self._llm_model_name,
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    **kwargs
                )
            else:
                # Para modelos locales, intentar usar litellm
                try:
                    from litellm import acompletion
                    resp = await acompletion(
                        model=self._llm_model_name,
                        messages=[
                            {"role": "system", "content": system_prompt or ""},
                            *([{"role": "user", "content": prompt}] if not history_messages else history_messages + [{"role": "user", "content": prompt}])
                        ],
                        **kwargs
                    )
                    return resp.choices[0].message.content
                except ImportError:
                    logger.error("❌ litellm no instalado. pip install litellm")
                    raise

        async def embedding_func(texts: list[str]) -> list[list[float]]:
            """Función de embeddings para LightRAG."""
            model = SentenceTransformer(self._model_name)
            embeddings = model.encode(texts, normalize_embeddings=True)
            return embeddings.tolist()

        self.rag = LightRAG(
            working_dir=self.working_dir,
            llm_model_func=llm_model_func,
            embedding_func=embedding_func,
            llm_model_name=self._llm_model_name,
            chunk_token_size=500,
            chunk_overlap_token_size=100,
            log_level=logging.INFO,
        )

        logger.info("✅ LightRAG inicializado correctamente")

    def insert_text(self, text: str) -> bool:
        """
        Inserta texto en el grafo de conocimiento.
        El sistema extrae automáticamente entidades y relaciones.

        Args:
            text: Texto a insertar

        Returns:
            True si tuvo éxito
        """
        try:
            import asyncio
            logger.info(f"📝 Insertando texto en LightRAG ({len(text)} chars)...")
            asyncio.run(self.rag.ainsert(text))
            logger.info("✅ Texto insertado en el grafo de conocimiento")
            return True
        except Exception as e:
            logger.error(f"❌ Error insertando texto: {e}")
            return False

    def insert_file(self, file_path: str) -> bool:
        """
        Lee e inserta un archivo de texto en el grafo.

        Args:
            file_path: Ruta al archivo (txt, md)

        Returns:
            True si tuvo éxito
        """
        try:
            path = Path(file_path)
            if not path.exists():
                logger.error(f"❌ Archivo no encontrado: {file_path}")
                return False

            if path.suffix.lower() in (".txt", ".md"):
                text = path.read_text(encoding="utf-8")
            else:
                # Intentar extraer texto
                try:
                    import pdfplumber
                    with pdfplumber.open(str(path)) as pdf:
                        text = "\n".join(p.extract_text() or "" for p in pdf.pages)
                except ImportError:
                    logger.error(f"❌ pdfplumber no disponible para extraer {file_path}")
                    return False

            if not text.strip():
                logger.warning(f"⚠️  Texto vacío en {file_path}")
                return False

            return self.insert_text(text)

        except Exception as e:
            logger.error(f"❌ Error procesando {file_path}: {e}")
            return False

    def insert_directory(self, dir_path: str) -> int:
        """
        Inserta todos los archivos de un directorio en el grafo.

        Args:
            dir_path: Directorio con archivos

        Returns:
            Número de archivos insertados exitosamente
        """
        dir_path = Path(dir_path)
        if not dir_path.exists():
            logger.error(f"❌ Directorio no encontrado: {dir_path}")
            return 0

        supported = ("*.txt", "*.md", "*.json")
        files = []
        for ext in supported:
            files.extend(sorted(dir_path.glob(ext)))

        if not files:
            logger.warning(f"⚠️  No se encontraron archivos soportados en {dir_path}")
            return 0

        success = 0
        for f in files:
            if self.insert_file(str(f)):
                success += 1
                logger.info(f"  ✅ {f.name} insertado")

        logger.info(f"📚 Total: {success}/{len(files)} archivos insertados en el grafo")
        return success

    def query(self, question: str, mode: str = "hybrid", top_k: int = 40,
              only_context: bool = False) -> str:
        """
        Consulta el grafo de conocimiento.

        Args:
            question: Pregunta del usuario
            mode: Modo de búsqueda
                - "local": entidades cercanas a los términos de la pregunta
                - "global": conceptos y temas generales
                - "hybrid": combinación de local y global (recomendado)
                - "naive": búsqueda vectorial tradicional
                - "mix": mezcla adaptativa de todos los modos
            top_k: Número de resultados a recuperar
            only_context: Si True, solo retorna el contexto sin generar respuesta

        Returns:
            Respuesta generada o contexto relevante
        """
        if self.rag is None:
            logger.error("❌ LightRAG no inicializado")
            return ""

        try:
            import asyncio

            param = QueryParam(
                mode=mode,
                only_need_context=only_context,
                top_k=top_k,
            )

            logger.info(f"🔍 Consultando LightRAG (mode={mode}, top_k={top_k})")
            logger.info(f"   Pregunta: {question[:100]}...")

            result = asyncio.run(self.rag.aquery(question, param=param))

            if only_context:
                logger.info(f"✅ Contexto recuperado: {len(result)} chars")
            else:
                logger.info(f"✅ Respuesta generada: {len(result)} chars")

            return result

        except Exception as e:
            logger.error(f"❌ Error en consulta: {e}")
            return f"Error: {e}"

    def stats(self):
        """Muestra estadísticas del grafo de conocimiento."""
        import json

        stats_path = Path(self.working_dir) / "graph_stats.json"
        if stats_path.exists():
            try:
                stats_data = json.loads(stats_path.read_text())
                print(f"📊 Estadísticas LightRAG:")
                print(f"   Working dir: {self.working_dir}")
                print(f"   Modelo embeddings: {self._model_name}")
                print(f"   Modelo LLM: {self._llm_model_name}")
                for key, value in stats_data.items():
                    print(f"   {key}: {value}")
            except Exception as e:
                print(f"   No se pudieron leer estadísticas: {e}")
        else:
            print(f"📊 LightRAG - {self.working_dir}")
            print(f"   (ejecuta index primero para generar estadísticas)")
            print(f"   Modelo embeddings: {self._model_name}")
            print(f"   Modelo LLM: {self._llm_model_name}")


def main():
    parser = argparse.ArgumentParser(
        description="LightRAG Engine — RAG con grafos de conocimiento",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modos de búsqueda:
  local   - entidades cercanas (preguntas específicas sobre datos concretos)
  global  - conceptos amplios (preguntas de resumen/tendencias)
  hybrid  - combinación local+global (recomendado para uso general)
  naive   - búsqueda vectorial tradicional
  mix     - mezcla adaptativa de todos los modos
        """
    )

    parser.add_argument("--working-dir", default=None,
                        help="Directorio de trabajo para persistencia")
    parser.add_argument("--llm-model", default=None,
                        help="Modelo LLM (default: gpt-4o-mini o LIGHTRAG_LLM_MODEL)")

    subparsers = parser.add_subparsers(dest="command", help="Comando")

    # Index
    idx_parser = subparsers.add_parser("index", help="Indexar un documento")
    idx_parser.add_argument("documento", help="Ruta al documento (txt, md, pdf)")

    # Index directory
    dir_parser = subparsers.add_parser("index-dir", help="Indexar directorio de documentos")
    dir_parser.add_argument("directorio", help="Directorio con documentos")

    # Query
    q_parser = subparsers.add_parser("query", help="Consultar el grafo de conocimiento")
    q_parser.add_argument("pregunta", help="Pregunta del usuario")
    q_parser.add_argument("--mode", choices=["local", "global", "hybrid", "naive", "mix"],
                          default="hybrid", help="Modo de búsqueda")
    q_parser.add_argument("--top-k", type=int, default=40,
                          help="Número de resultados a recuperar")
    q_parser.add_argument("--context-only", action="store_true",
                          help="Solo mostrar contexto, sin generar respuesta")

    # Stats
    subparsers.add_parser("stats", help="Estadísticas del grafo")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if not HAS_LIGHTRAG:
        print("❌ lightrag-hku no instalado.")
        print("   pip install lightrag-hku")
        sys.exit(1)

    try:
        engine = LightRagEngine(
            working_dir=args.working_dir,
            llm_model_name=args.llm_model,
        )
    except ImportError as e:
        print(f"❌ {e}")
        sys.exit(1)

    if args.command == "index":
        success = engine.insert_file(args.documento)
        if success:
            print(f"✅ Documento indexado en el grafo de conocimiento")
        else:
            print(f"❌ Error indexando documento")
            sys.exit(1)

    elif args.command == "index-dir":
        count = engine.insert_directory(args.directorio)
        print(f"📚 {count} archivos indexados en el grafo de conocimiento")

    elif args.command == "query":
        result = engine.query(
            args.pregunta,
            mode=args.mode,
            top_k=args.top_k,
            only_context=args.context_only,
        )
        if result:
            if args.context_only:
                print(f"\n📄 Contexto recuperado:\n{result}")
            else:
                print(f"\n💬 Respuesta:\n{result}")

    elif args.command == "stats":
        engine.stats()


if __name__ == "__main__":
    main()
