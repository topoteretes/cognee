import argparse
import asyncio
import logging
from contextlib import contextmanager

from cognee.cli import DEFAULT_DOCS_URL
from cognee.cli.exceptions import CliCommandException
from cognee.cli.reference import SupportsCliCommand
import cognee.cli.echo as fmt


@contextmanager
def _suppressed_error_logs():
    """The checks below log failures at ERROR with full tracebacks (meant for
    server logs); doctor reports the same failures as ✗ lines, so silence the
    duplicate traceback dumps for the duration of the check."""
    logging.disable(logging.ERROR)
    try:
        yield
    finally:
        logging.disable(logging.NOTSET)


class DoctorCommand(SupportsCliCommand):
    """Diagnose cognee configuration and local services before first use."""

    command_string = "doctor"
    help_string = "Diagnose configuration and local services (config traps, databases, providers)"
    docs_url = DEFAULT_DOCS_URL
    description = """
The `doctor` command runs preflight diagnostics and prints a pass/fail report:

1. Provider configuration consistency — catches the only-LLM-or-only-embeddings
   trap where the unconfigured side silently defaults to OpenAI and breaks
   minutes into the first ingestion.
2. Local services — relational, vector and graph databases plus file storage
   (same checks as the /health endpoint).
3. With --probe: live LLM and embedding round-trips (makes network calls and
   may incur token costs).

Exits non-zero when any check fails, so it can gate CI and setup scripts.
"""

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--probe",
            action="store_true",
            help="Also run live LLM and embedding connectivity probes (network calls; may cost tokens)",
        )

    def execute(self, args: argparse.Namespace) -> None:
        failures = asyncio.run(self._run(probe=args.probe))
        if failures:
            raise CliCommandException(
                f"doctor found {failures} problem{'s' if failures != 1 else ''} (see above)",
                error_code=1,
            )
        fmt.success("\nAll checks passed. cognee is ready to use.")

    async def _run(self, probe: bool) -> int:
        failures = 0

        # --- 1. Provider configuration consistency (zero network) ---
        fmt.bold("Configuration")
        from cognee.infrastructure.llm.config import get_llm_context_config
        from cognee.infrastructure.databases.vector.embeddings.config import (
            get_embedding_context_config,
        )
        from cognee.modules.preflight import check_provider_config

        llm_config = get_llm_context_config()
        embedding_config = get_embedding_context_config()
        fmt.echo(
            f"  LLM:        provider={llm_config.llm_provider} model={llm_config.llm_model} "
            f"api_key={'set' if (llm_config.llm_api_key or '').strip() else 'NOT SET'}"
        )
        fmt.echo(
            f"  Embeddings: provider={embedding_config.embedding_provider} "
            f"model={embedding_config.embedding_model} "
            f"api_key={'set' if (embedding_config.embedding_api_key or '').strip() else 'NOT SET'} "
            f"dimensions={embedding_config.embedding_dimensions}"
        )

        problems = check_provider_config(llm_config, embedding_config)
        for problem in problems:
            fmt.error(f"  ✗ {problem}")
            failures += 1
        if not problems:
            fmt.echo("  ✓ LLM/embedding provider configuration is consistent")

        # --- 2. Local services (same checks as the /health endpoint) ---
        fmt.bold("\nServices")
        from cognee.api.v1.health.health import HealthStatus, health_checker

        with _suppressed_error_logs():
            health = await health_checker.get_health_status(detailed=False)
        for name, component in health.components.items():
            if component.status == HealthStatus.HEALTHY:
                fmt.echo(f"  ✓ {name} ({component.provider}, {component.response_time_ms}ms)")
            else:
                fmt.error(f"  ✗ {name} ({component.provider}): {component.details}")
                failures += 1
                if name == "relational_db":
                    fmt.note(
                        "A missing local database is created on first use — run any "
                        "`cognee-cli remember` command, or check DB_* settings if you "
                        "configured Postgres."
                    )

        # --- 3. Optional live provider probes ---
        if probe:
            fmt.bold("\nProviders (live probes)")
            from cognee.infrastructure.llm.utils import (
                test_embedding_connection,
                test_llm_connection,
            )

            try:
                with _suppressed_error_logs():
                    await test_llm_connection()
                fmt.echo("  ✓ LLM round-trip succeeded")
            except Exception as error:
                fmt.error(f"  ✗ LLM round-trip failed: {error}")
                failures += 1

            try:
                with _suppressed_error_logs():
                    dimensions = await test_embedding_connection()
                fmt.echo(f"  ✓ Embedding round-trip succeeded (dimensions={dimensions})")
            except Exception as error:
                fmt.error(f"  ✗ Embedding round-trip failed: {error}")
                failures += 1
        else:
            fmt.echo("")
            fmt.note("Run `cognee-cli doctor --probe` to also test live LLM/embedding calls.")

        return failures
