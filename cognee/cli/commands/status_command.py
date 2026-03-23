import argparse
import asyncio
from typing import Optional

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException


class StatusCommand(SupportsCliCommand):
    command_string = "status"
    help_string = "Check if Cognee is configured and ready"
    docs_url = DEFAULT_DOCS_URL
    description = """
Preflight check: shows whether API keys, databases, and datasets are configured.

Useful for agents to verify the system is ready before running commands.
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        pass

    def execute(self, args: argparse.Namespace) -> Optional[dict]:
        try:
            from cognee.infrastructure.llm.config import get_llm_config
            from cognee.infrastructure.databases.vector.embeddings.config import (
                get_embedding_config,
            )
            from cognee.infrastructure.databases.graph.config import get_graph_config
            from cognee.infrastructure.databases.vector.config import get_vectordb_config
            from cognee.infrastructure.databases.relational.config import get_relational_config

            checks = {}

            # LLM
            llm = get_llm_config()
            has_key = bool(llm.llm_api_key)
            checks["llm"] = {
                "provider": llm.llm_provider,
                "model": llm.llm_model,
                "api_key_set": has_key,
            }
            fmt.echo(
                f"  LLM: {llm.llm_provider}/{llm.llm_model} — key {'set' if has_key else 'MISSING'}"
            )

            # Embedding
            emb = get_embedding_config()
            emb_key = bool(emb.embedding_api_key)
            checks["embedding"] = {
                "provider": emb.embedding_provider,
                "model": emb.embedding_model,
                "api_key_set": emb_key,
            }
            fmt.echo(
                f"  Embedding: {emb.embedding_provider}/{emb.embedding_model} — key {'set' if emb_key else 'MISSING'}"
            )

            # Databases
            graph = get_graph_config()
            vec = get_vectordb_config()
            rel = get_relational_config()
            checks["databases"] = {
                "graph": graph.graph_database_provider,
                "vector": vec.vector_db_provider,
                "relational": rel.db_provider,
            }
            fmt.echo(f"  Graph DB: {graph.graph_database_provider}")
            fmt.echo(f"  Vector DB: {vec.vector_db_provider}")
            fmt.echo(f"  Relational DB: {rel.db_provider}")

            # Datasets
            try:
                from cognee.modules.users.methods import get_default_user
                import cognee

                async def _count():
                    user = await get_default_user()
                    ds = await cognee.datasets.list_datasets(user=user)
                    return len(ds) if ds else 0

                count = asyncio.run(_count())
                checks["datasets"] = count
                fmt.echo(f"  Datasets: {count}")
            except Exception:
                checks["datasets"] = "unavailable"
                fmt.warning("  Datasets: could not query")

            # Overall readiness
            ready = has_key
            checks["ready"] = ready
            if ready:
                fmt.success("System ready.")
            else:
                fmt.warning("System not ready — LLM API key is missing.")

            return checks

        except Exception as e:
            raise CliCommandException(f"Status check failed: {e}", error_code=1) from e
