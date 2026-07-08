import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL


@dataclass
class CheckResult:
    check_id: str
    status: str  # ok | note | fail | skip
    summary: str
    fix: Optional[str] = None
    details: List[str] = field(default_factory=list)


def _mask_key(key: str) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return key[:2] + "****"
    return key[:3] + "****" + key[-4:]


class DoctorCommand(SupportsCliCommand):
    command_string = "doctor"
    help_string = "Check that your setup is ready (config, keys, databases, graph)"
    docs_url = DEFAULT_DOCS_URL
    description = """
Check that your cognee setup is ready to build and search memory.

Runs local checks first (Python version, configuration, storage), then one
cheap call each to your LLM and embedding providers — so a broken key or an
unreachable endpoint shows up here, in seconds, instead of minutes into your
first cognify.

Exit code 0 when healthy (notes are fine), 1 when something needs fixing.
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "-v",
            "--verbose",
            action="store_true",
            help="Show extra details for every check (paths, versions, endpoints)",
        )
        parser.add_argument(
            "--offline", action="store_true", help="Skip network checks (LLM/embedding pings)"
        )
        parser.add_argument(
            "--json", action="store_true", help="Machine-readable output for CI gates"
        )

    def execute(self, args: argparse.Namespace) -> None:
        checks = asyncio.run(self._run_checks(offline=args.offline))

        if args.json:
            payload = {
                "checks": [
                    {
                        "id": check.check_id,
                        "status": check.status,
                        "summary": check.summary,
                        "fix": check.fix,
                        "details": check.details,
                    }
                    for check in checks
                ],
                "summary": {
                    "total": len(checks),
                    "failures": sum(1 for c in checks if c.status == "fail"),
                    "notes": sum(1 for c in checks if c.status == "note"),
                },
            }
            sys.stdout.write(json.dumps(payload, indent=2) + "\n")
        else:
            self._render(checks, verbose=args.verbose)

        if any(check.status == "fail" for check in checks):
            raise SystemExit(1)

    # ------------------------------------------------------------------ #

    def _render(self, checks: List[CheckResult], verbose: bool) -> None:
        from cognee.cli import ui

        caps = ui.detect_caps()
        style = ui.Style(caps.color and caps.stdout_tty)
        # doctor prints to stdout, so probe stdout — caps.unicode reflects
        # stderr, and the two can disagree (e.g. stdout piped to a file).
        unicode_ok = ui._supports_unicode(sys.stdout)
        glyphs = ui.Glyphs(unicode_ok)

        marks = {
            "ok": style.green(f"[{glyphs.ok}]" if unicode_ok else "[ok]"),
            "note": style.yellow("[!]"),
            "fail": style.red(f"[{glyphs.fail}]" if unicode_ok else "[FAIL]"),
            "skip": style.dim("[-]"),
        }

        header = style.bold("cognee doctor")
        if not verbose:
            header += style.dim(" — checking your setup (details: doctor -v)")
        sys.stdout.write(header + "\n\n")

        for check in checks:
            sys.stdout.write(f" {marks[check.status]} {check.summary}\n")
            if check.fix and check.status in ("note", "fail"):
                sys.stdout.write(f"     {check.fix}\n")
            if verbose:
                for detail in check.details:
                    sys.stdout.write(style.dim(f"     {glyphs.bullet} {detail}") + "\n")

        failures = sum(1 for c in checks if c.status == "fail")
        notes = sum(1 for c in checks if c.status == "note")
        sys.stdout.write("\n")
        if failures:
            sys.stdout.write(
                f"{style.yellow('!')} Doctor found issues in {failures} "
                f"categor{'ies' if failures != 1 else 'y'}.\n"
            )
            first_fix = next((c.fix for c in checks if c.status == "fail" and c.fix), None)
            if first_fix:
                sys.stdout.write(f"  Start here: {first_fix}\n")
        else:
            checked = sum(1 for c in checks if c.status != "skip")
            note_part = f", {notes} note{'s' if notes != 1 else ''}" if notes else ""
            sys.stdout.write(
                f"{style.green(glyphs.bullet)} {checked} checks{note_part}. "
                + style.bold("Your setup is ready to cognify.")
                + "\n"
            )

        if verbose:
            hints_state = (
                "off (COGNEE_NO_HINTS set)"
                if os.environ.get("COGNEE_NO_HINTS")
                else ("on (set COGNEE_NO_HINTS=1 to disable)")
            )
            sys.stdout.write(style.dim(f"\nHints: {hints_state}") + "\n")
            try:
                from cognee.shared.logging_utils import get_log_file_location

                log_file = get_log_file_location()
                if log_file:
                    sys.stdout.write(style.dim(f"Logs:  {log_file}") + "\n")
            except Exception:
                pass

    # ------------------------------------------------------------------ #

    async def _run_checks(self, offline: bool) -> List[CheckResult]:
        checks: List[CheckResult] = []
        checks.append(self._check_python())
        checks.append(self._check_config_source())
        llm_check, llm_ok = self._check_llm_settings()
        checks.append(llm_check)
        checks.append(self._check_embedding_settings())
        checks.append(self._check_storage())
        databases_check, databases_initialized = self._check_databases()
        checks.append(databases_check)

        if offline:
            checks.append(CheckResult("llm_reachable", "skip", "LLM — not checked (--offline)"))
            checks.append(
                CheckResult("embeddings_reachable", "skip", "Embeddings — not checked (--offline)")
            )
        elif not llm_ok:
            checks.append(
                CheckResult("llm_reachable", "skip", "LLM — not checked (fix the key first)")
            )
            checks.append(
                CheckResult(
                    "embeddings_reachable", "skip", "Embeddings — not checked (fix the key first)"
                )
            )
        else:
            checks.append(await self._check_llm_reachable())
            checks.append(await self._check_embeddings_reachable())

        checks.append(await self._check_graph(databases_initialized))
        return checks

    def _check_python(self) -> CheckResult:
        version = sys.version_info
        pretty = f"{version.major}.{version.minor}.{version.micro}"
        if (3, 10) <= (version.major, version.minor) <= (3, 14):
            return CheckResult("python", "ok", f"Python {pretty}", details=[sys.executable])
        return CheckResult(
            "python",
            "fail",
            f"Python {pretty} — cognee supports 3.10 to 3.14",
            fix="Install a supported Python and recreate your virtualenv",
        )

    def _check_config_source(self) -> CheckResult:
        env_file = os.path.join(os.getcwd(), ".env")
        cognee_vars = sorted(
            var
            for var in os.environ
            if var.startswith(
                ("LLM_", "EMBEDDING_", "COGNEE_", "GRAPH_DATABASE_", "VECTOR_", "DB_")
            )
        )
        details = [f"environment variables: {', '.join(cognee_vars) or 'none'}"]
        if os.path.isfile(env_file):
            try:
                with open(env_file, encoding="utf-8") as handle:
                    settings = [
                        line
                        for line in handle
                        if line.strip() and not line.strip().startswith("#") and "=" in line
                    ]
                count = len(settings)
            except OSError:
                count = 0
            details.insert(0, env_file)
            return CheckResult(
                "config",
                "ok",
                f"Config .env found ({count} setting{'s' if count != 1 else ''})",
                details=details,
            )
        if cognee_vars:
            count = len(cognee_vars)
            return CheckResult(
                "config",
                "ok",
                f"Config from environment ({count} variable{'s' if count != 1 else ''})",
                details=details,
            )
        return CheckResult(
            "config",
            "note",
            "No .env file and no cognee environment variables — defaults in effect",
            fix="Copy .env.template to .env and set LLM_API_KEY",
            details=details,
        )

    def _check_llm_settings(self) -> tuple:
        from cognee.cli.preflight import _KEY_REQUIRED, _PLACEHOLDER_KEYS

        try:
            from cognee.infrastructure.llm.config import get_llm_config

            config = get_llm_config()
        except Exception as error:
            return (
                CheckResult(
                    "llm_config",
                    "fail",
                    "LLM configuration could not be loaded",
                    fix=str(error).splitlines()[0][:120],
                ),
                False,
            )

        provider = (config.llm_provider or "").lower()
        model = config.llm_model or ""
        key = (config.llm_api_key or "").strip()
        details = [f"provider: {provider}", f"model: {model}"]
        if config.llm_endpoint:
            details.append(f"endpoint: {config.llm_endpoint}")

        known_providers = {
            "openai",
            "ollama",
            "anthropic",
            "custom",
            "gemini",
            "mistral",
            "azure",
            "bedrock",
            "llama_cpp",
            "lm_studio",
        }
        if provider not in known_providers:
            return (
                CheckResult(
                    "llm_config",
                    "fail",
                    f"Unknown LLM provider '{provider}'",
                    fix=f"Set LLM_PROVIDER to one of: {', '.join(sorted(known_providers))}",
                    details=details,
                ),
                False,
            )

        if provider in _KEY_REQUIRED:
            if key.lower() in _PLACEHOLDER_KEYS:
                what = "looks like a placeholder" if key else "is not set"
                return (
                    CheckResult(
                        "llm_config",
                        "fail",
                        f"LLM {model} — LLM_API_KEY {what}",
                        fix="export LLM_API_KEY=sk-...",
                        details=details,
                    ),
                    False,
                )
            summary = f"LLM {model} — key {_mask_key(key)}"
        else:
            summary = f"LLM {model} (no key needed for {provider})"

        return CheckResult("llm_config", "ok", summary, details=details), True

    def _check_embedding_settings(self) -> CheckResult:
        from cognee.cli.preflight import _embedding_env_configured

        explicit = _embedding_env_configured()
        try:
            from cognee.infrastructure.llm.config import get_llm_config

            llm_config = get_llm_config()
        except Exception:
            return CheckResult(
                "embeddings", "skip", "Embeddings — not checked (LLM config unavailable)"
            )

        if explicit:
            try:
                from cognee.infrastructure.databases.vector.embeddings.config import (
                    get_embedding_config,
                )

                config = get_embedding_config()
                return CheckResult(
                    "embeddings",
                    "ok",
                    f"Embeddings {config.embedding_model or config.embedding_provider}",
                    details=[f"provider: {config.embedding_provider} (explicit)"],
                )
            except Exception as error:
                return CheckResult(
                    "embeddings",
                    "fail",
                    "Embedding configuration could not be loaded",
                    fix=str(error).splitlines()[0][:120],
                )

        try:
            from cognee.infrastructure.databases.vector.embeddings.derive_embedding_settings import (
                derive_embedding_settings,
            )

            derived = derive_embedding_settings(
                llm_config.llm_provider, llm_config.llm_endpoint or None, llm_config.llm_api_key
            )
        except Exception as error:
            return CheckResult(
                "embeddings",
                "fail",
                f"Embeddings can't be derived from LLM provider '{llm_config.llm_provider}'",
                fix=str(error).splitlines()[0][:160],
            )

        if derived is None:
            return CheckResult(
                "embeddings",
                "note",
                f"Embeddings default in effect for provider '{llm_config.llm_provider}'",
                fix="Set EMBEDDING_PROVIDER / EMBEDDING_MODEL explicitly",
            )
        return CheckResult(
            "embeddings",
            "ok",
            f"Embeddings {derived['model'] or derived['provider']} (derived from LLM provider)",
            details=[f"provider: {derived['provider']}"],
        )

    def _check_storage(self) -> CheckResult:
        try:
            from cognee.base_config import get_base_config

            base = get_base_config()
            roots = [base.data_root_directory, base.system_root_directory]
            details = [str(root) for root in roots]
            for root in roots:
                if not root or str(root).startswith("s3://"):
                    continue
                os.makedirs(root, exist_ok=True)
                probe = os.path.join(root, ".cognee_write_probe")
                with open(probe, "w") as handle:
                    handle.write("ok")
                os.remove(probe)
            home = os.path.commonprefix([str(r) for r in roots if r]) or str(roots[0])
            return CheckResult("storage", "ok", f"Storage {home} — writable", details=details)
        except OSError as error:
            return CheckResult(
                "storage",
                "fail",
                "Storage directory is not writable",
                fix=f"Check permissions: {error}",
            )
        except Exception as error:
            return CheckResult(
                "storage", "fail", "Storage configuration failed", fix=str(error)[:120]
            )

    def _check_databases(self) -> tuple:
        try:
            from cognee.base_config import get_base_config
            from cognee.infrastructure.databases.graph.config import get_graph_config
            from cognee.infrastructure.databases.relational.config import get_relational_config
            from cognee.infrastructure.databases.vector.config import get_vectordb_config

            relational = get_relational_config().db_provider
            vector = get_vectordb_config().vector_db_provider
            graph = get_graph_config().graph_database_provider
            providers = f"{relational} · {vector} · {graph}"

            databases_path = os.path.join(str(get_base_config().system_root_directory), "databases")
            initialized = os.path.isdir(databases_path) and bool(os.listdir(databases_path))
            state = "initialized" if initialized else "will be created on first use"
            return (
                CheckResult(
                    "databases",
                    "ok",
                    f"Databases {providers} — {state}",
                    details=[databases_path],
                ),
                initialized,
            )
        except Exception as error:
            return (
                CheckResult(
                    "databases", "fail", "Database configuration failed", fix=str(error)[:120]
                ),
                False,
            )

    async def _check_llm_reachable(self) -> CheckResult:
        try:
            from cognee.infrastructure.llm.utils import test_llm_connection

            started = time.monotonic()
            await asyncio.wait_for(test_llm_connection(), timeout=35)
            latency_ms = int((time.monotonic() - started) * 1000)
            return CheckResult("llm_reachable", "ok", f"LLM reachable ({latency_ms} ms)")
        except Exception as error:
            message = str(error).splitlines()[0][:160] if str(error) else type(error).__name__
            return CheckResult(
                "llm_reachable",
                "fail",
                "LLM is not reachable",
                fix=message,
            )

    async def _check_embeddings_reachable(self) -> CheckResult:
        try:
            from cognee.infrastructure.llm.utils import test_embedding_connection

            started = time.monotonic()
            dimensions = await asyncio.wait_for(test_embedding_connection(), timeout=35)
            latency_ms = int((time.monotonic() - started) * 1000)
            return CheckResult(
                "embeddings_reachable",
                "ok",
                f"Embeddings reachable ({latency_ms} ms)",
                details=[f"vector dimensions: {dimensions}"],
            )
        except Exception as error:
            message = str(error).splitlines()[0][:160] if str(error) else type(error).__name__
            return CheckResult(
                "embeddings_reachable",
                "fail",
                "Embeddings are not reachable",
                fix=message,
            )

    async def _check_graph(self, databases_initialized: bool) -> CheckResult:
        """Answered from the relational pipeline-run table, NOT a graph-engine
        probe: under multi-tenant access control (the default) each dataset
        has its own graph database, so a bare engine probe would look at the
        wrong one."""
        if not databases_initialized:
            return CheckResult(
                "graph",
                "skip",
                "Graph — not checked (databases not initialized yet)",
            )
        try:
            from cognee.cli.empty_state import check_memory_state
            from cognee.cli.user_resolution import resolve_cli_user

            user = await resolve_cli_user(None)
            state, dataset_name, doc_count = await check_memory_state(user)
            if state == "empty":
                return CheckResult(
                    "graph",
                    "note",
                    "Memory is empty",
                    fix="Run: cognee-cli add <file>, then cognee-cli cognify",
                )
            if state == "not_cognified":
                where = f" in {dataset_name}" if dataset_name else ""
                return CheckResult(
                    "graph",
                    "note",
                    f"Data added{where} ({doc_count} document(s)) but no graph built yet",
                    fix="Run: cognee-cli cognify",
                )
            return CheckResult("graph", "ok", "Knowledge graph built")
        except Exception as error:
            return CheckResult(
                "graph",
                "note",
                "Graph — could not be checked",
                fix=str(error).splitlines()[0][:120],
            )
