"""CLI commands for the Cognee Hermes memory plugin."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from . import catalog, code_graph, recap, runtime_env, update_check
from .config import (
    DEFAULT_LOCAL_PORT,
    LOCAL_COGNEE_REQUIREMENT,
    RECALL_MODE_PINNED,
    config_path,
    configured_dataset_name,
    effective_recall_mode,
    load_config,
    recall_mode_is_set,
    resolve_local_roots,
)


def _provider_active() -> bool:
    try:
        from hermes_cli.config import cfg_get
        from hermes_cli.config import load_config as load_hermes_config

        config = load_hermes_config()
        return cfg_get(config, "memory", "provider") == "cognee"
    except Exception:
        return False


def _installed_plugin_version() -> str:
    """The version of the plugin copy this code runs from (its plugin.yaml)."""
    manifest = Path(__file__).resolve().parents[1] / "plugin.yaml"
    try:
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.startswith("version:"):
                return line.split(":", 1)[1].strip().strip("\"'")
    except OSError:
        pass
    return ""


def _pip_package_version() -> str:
    """The pip-installed package version, if the package is in this env."""
    try:
        from importlib.metadata import version

        return version("cognee-integration-hermes-agent")
    except Exception:
        return ""


def _update_hint(cfg, *, force: bool = False) -> str:
    """A one-line nudge when PyPI has a newer release, else "". Never raises."""
    name = catalog.catalog_name()
    if name:
        return f"managed by the Hermes catalog — run `hermes plugins update {name}`"
    if not cfg.get("update_check", True):
        return ""
    pip_version = _pip_package_version()
    if not pip_version:
        return ""
    latest = update_check.latest_published_version(
        interval=float(cfg.get("update_check_interval") or 3600), force=force
    )
    if not update_check.is_newer(latest, pip_version):
        return ""
    return (
        f"update available: {pip_version} → {latest} — run "
        f"`pip install -U {update_check.PYPI_PACKAGE}` then `cognee-hermes-install`"
    )


def _print_status(args) -> None:
    cfg = load_config()
    path = config_path()
    plugin_version = _installed_plugin_version()
    pip_version = _pip_package_version()
    print("\nCognee memory")
    print("-" * 40)
    print(f"  Active provider: {'yes' if _provider_active() else 'no'}")
    print(f"  Local runtime:   {_runtime_status()}")
    print(f"  Plugin version:  {plugin_version or '(unknown)'}")
    print(f"  Mode:            {'remote' if cfg.get('service_url') else 'local'}")
    print(f"  Dataset:         {cfg.get('dataset')}")
    print(f"  Recall:          {_recall_status(cfg)}")
    print(f"  Config:          {path or '(unknown)'}")
    print(f"  Service URL:     {cfg.get('service_url') or '(none)'}")
    print(f"  LLM key:         {'set' if cfg.get('llm_api_key') else 'missing'}")
    print(f"  API key:         {'set' if cfg.get('api_key') else 'missing'}")
    print(f"  Improve on end:  {cfg.get('improve_on_end')}")
    if (
        not catalog.catalog_name()
        and pip_version
        and plugin_version
        and pip_version != plugin_version
    ):
        # Hermes runs the copy under HERMES_HOME/plugins, so `pip install -U`
        # alone changes nothing until the installer refreshes that copy.
        print(
            f"  Update:          pip package is {pip_version} but this installed copy "
            f"is {plugin_version} — run `cognee-hermes-install` to update"
        )
    else:
        hint = _update_hint(cfg, force=getattr(args, "check_updates", False))
        if hint:
            print(f"  Update:          {hint}")
    print()


def _recall_status(cfg) -> str:
    """The configured recall mode in effect, and why when no mode is set."""
    mode = effective_recall_mode(cfg)
    if recall_mode_is_set(cfg) or mode != RECALL_MODE_PINNED:
        return mode
    return (
        f"pinned to {configured_dataset_name(cfg.get('dataset'))} (the configured dataset, no "
        "recall mode set; `recall_mode: auto` in cognee.json lets Cognee pick)"
    )


def _print_version(args) -> None:
    cfg = load_config()
    plugin_version = _installed_plugin_version()
    pip_version = _pip_package_version()
    print(f"cognee-integration-hermes-agent {plugin_version or pip_version or '(unknown)'}")
    if (
        not catalog.catalog_name()
        and plugin_version
        and pip_version
        and plugin_version != pip_version
    ):
        print(
            f"  installed plugin copy is {plugin_version} — run `cognee-hermes-install` "
            "to refresh it from the pip package"
        )
    hint = _update_hint(cfg, force=getattr(args, "check_updates", False))
    if hint:
        print(f"  {hint}")


def _connect_backend():
    """A connected HttpBackend for one-shot CLI operations.

    Mirrors the provider's connection modes minus embedded (these operations
    are HTTP-only): a set service_url is used as-is, otherwise the shared
    local server is ensured — booted if nobody has yet — exactly as a session
    would. Registration keeps the server's idle watchdog honest for the
    duration; the caller must close().
    """
    from .http_backend import HttpBackend
    from .server_bootstrap import ensure_local_server

    cfg = load_config()
    backend = HttpBackend(agent_session_name="hermes-cli")
    service_url = str(cfg.get("service_url") or "")
    if service_url:
        backend.connect(url=service_url, api_key=str(cfg.get("api_key") or ""), timeout=30)
        return backend
    data_root, system_root = resolve_local_roots(cfg)
    local_url = ensure_local_server(
        int(cfg.get("local_port") or DEFAULT_LOCAL_PORT),
        data_root=data_root,
        system_root=system_root,
        boot_timeout=float(cfg.get("server_boot_timeout") or 600),
    )
    backend.connect(url=local_url, api_key="", timeout=30)
    return backend


# -- plugin agent keys -------------------------------------------------------
#
# Every machine a cognee plugin (Claude Code, Codex, ...) runs on holds its own
# API key for that plugin's agent (cognee#6113). The server caps them; when an
# agent is full of keys still in use, a new machine is refused one and runs on
# the user's personal key until an old machine's key is revoked. These commands
# are how a Hermes user frees a slot — for any plugin agent of their user, since
# Hermes itself registers as the principal and provisions no agent.

#: Plugins whose agents are listed when the server's own listing cannot be read.
FALLBACK_PLUGIN_KEYS = ("claude-code", "codex", "opencode", "openclaw", "mcp", "api", "desktop")


def _known_plugin_keys(backend) -> list[str]:
    try:
        status = backend._request("GET", "/api/v1/integrations/status", timeout=20)
        rows = (status.get("plugins") if isinstance(status, dict) else None) or []
        keys = [str(r.get("key") or "") for r in rows if isinstance(r, dict)]
        keys = [k for k in keys if k]
        if keys:
            return keys
    except Exception:
        pass
    return list(FALLBACK_PLUGIN_KEYS)


def _plugin_keys(backend, plugin: str) -> list[dict]:
    """``plugin``'s agent keys, or [] when the server has no key routes for it."""
    from .http_backend import CogneeHttpError

    try:
        body = backend._request("GET", f"/api/v1/integrations/plugins/{plugin}/keys", timeout=20)
    except CogneeHttpError as exc:
        if exc.status in (404, 405):
            return []
        raise
    rows = body.get("keys") if isinstance(body, dict) else None
    return [
        {
            "plugin": plugin,
            "id": str(r.get("id") or ""),
            "label": str(r.get("label") or ""),
            "created_at": str(r.get("created_at") or r.get("createdAt") or ""),
            "last_used_at": str(r.get("last_used_at") or r.get("lastUsedAt") or ""),
        }
        for r in rows or []
        if isinstance(r, dict)
    ]


def sort_keys_by_last_use(rows: list[dict]) -> list[dict]:
    """Most recently used first; never-used keys last, newest of those first."""
    return sorted(
        rows,
        key=lambda r: (1, r["last_used_at"]) if r["last_used_at"] else (0, r["created_at"]),
        reverse=True,
    )


def _fmt_stamp(value: str) -> str:
    return value.replace("T", " ")[:16] if value else "never"


def _run_keys_list(args) -> int:
    backend = _connect_backend()
    try:
        plugin = str(getattr(args, "plugin", "") or "")
        rows = []
        for key in [plugin] if plugin else _known_plugin_keys(backend):
            rows.extend(_plugin_keys(backend, key))
        rows = sort_keys_by_last_use(rows)
    finally:
        backend.close()
    if getattr(args, "json", False):
        print(json.dumps({"ok": True, "keys": rows}))
        return 0
    if not rows:
        print("No plugin agent keys" + (f" for {plugin!r}." if plugin else "."))
        return 0
    print(f"{'#':>2}  {'last used':<16}  {'created':<16}  {'plugin':<12}  id")
    for n, row in enumerate(rows, 1):
        print(
            f"{n:>2}  {_fmt_stamp(row['last_used_at']):<16}  {_fmt_stamp(row['created_at']):<16}  "
            f"{row['plugin']:<12}  {row['id']}"
        )
    print("\nRevoke one: hermes cognee keys revoke <id> --plugin <plugin>")
    return 0


def _run_keys_revoke(args) -> int:
    from .http_backend import CogneeHttpError

    plugin, key_id = str(args.plugin), str(args.key_id)
    backend = _connect_backend()
    try:
        backend._request(
            "DELETE", f"/api/v1/integrations/plugins/{plugin}/keys/{key_id}", timeout=20
        )
    except CogneeHttpError as exc:
        if exc.status == 404:
            print(
                f"No key {key_id} on the {plugin!r} agent (already revoked, or another plugin's)."
            )
            return 3
        raise
    finally:
        backend.close()
    print(f"Revoked key {key_id} of the {plugin!r} agent.")
    return 0


def _run_index_repo(args) -> int:
    """Index one repository into a deterministic code graph (cognee >= 1.5.4)."""
    spec = code_graph.canonical_spec(str(args.repo))
    if not code_graph.is_remote_repo(spec) and not Path(spec).is_dir():
        print(f"Error: {args.repo!r} is not a directory or a recognized git URL.")
        return 1
    dataset = str(getattr(args, "dataset", "") or "") or code_graph.default_code_dataset(spec)
    index_vectors = bool(getattr(args, "index_vectors", False))
    wait_seconds = float(getattr(args, "wait", 0.0) or 0.0)

    try:
        backend = _connect_backend()
    except Exception as exc:
        print(f"Error: could not reach a cognee server: {exc}")
        return 1
    try:
        result = backend.index_repository(
            repo=spec, dataset=dataset, index_vectors=index_vectors, timeout=120
        )
        status = str(result.get("status") or "submitted")
        print(f"Indexing {spec}")
        print(f"  dataset: {dataset}")
        print(f"  status:  {status}")
        dataset_id = str(result.get("dataset_id") or "")
        if wait_seconds > 0 and dataset_id:
            outcome = _poll_code_graph(backend, dataset_id, wait_seconds)
            print(f"  outcome: {outcome}")
            status = outcome
        elif wait_seconds > 0:
            print("  outcome: unknown (the server returned no dataset_id to poll)")
        code_graph.record_index(spec, dataset, index_vectors=index_vectors, status=status)
        print(
            "Registered for code recall. Query it with the cognee_code_search tool "
            "in a Hermes session."
        )
        return 0
    except Exception as exc:
        print(f"Error: indexing failed: {exc}")
        return 1
    finally:
        try:
            backend.close(timeout=5)
        except Exception:
            pass


def _run_recap(args) -> int:
    """``hermes cognee standup|digest|timeline``: the recap skeleton for this dataset."""
    from .provider import _safe_dataset_name

    cfg = load_config()
    dataset = _safe_dataset_name(str(getattr(args, "dataset", "") or cfg.get("dataset") or ""))
    try:
        backend = _connect_backend()
    except Exception as exc:
        print(f"Error: could not reach a cognee server: {exc}", file=sys.stderr)
        return 1
    try:
        result = recap.run_recap(
            recap.BackendRecapApi(backend),
            kind=args.cognee_command,
            dataset_ids=recap.resolve_dataset_ids(backend, [dataset]),
            datasets=[dataset],
            topic=str(getattr(args, "topic", "") or ""),
            since=str(args.since or ""),
            projects=[p.strip() for p in (args.projects or "").split(",") if p.strip()],
            all_sessions=bool(args.all_sessions),
            max_sessions=int(args.max_sessions),
            max_learnings=int(args.max_learnings),
            session_prefixes=(f"{cfg.get('session_prefix') or 'hermes'}_",),
        )
    except Exception as exc:
        print(f"Error: no recap: {exc}", file=sys.stderr)
        return 1
    finally:
        try:
            backend.close(timeout=5)
        except Exception:
            pass
    for warning in result["warnings"]:
        print(f"[cognee recap] {warning}", file=sys.stderr)
    if args.json:
        print(json.dumps(result["json"], indent=2, default=str))
    else:
        print(result["markdown"])
    return 0


def _poll_code_graph(backend, dataset_id: str, deadline_seconds: float) -> str:
    """Poll the code_graph_pipeline until completed/errored/timeout."""
    deadline = time.monotonic() + max(0.0, deadline_seconds)
    while True:
        status = ""
        try:
            status = backend.dataset_pipeline_status(dataset_id=dataset_id, timeout=10)
        except Exception:
            pass
        if status.endswith("COMPLETED"):
            return "completed"
        if status.endswith("ERRORED"):
            return "errored"
        if time.monotonic() >= deadline:
            return "timeout"
        time.sleep(3.0)


def _print_config(args) -> None:
    cfg = dict(load_config())
    for key in ("llm_api_key", "api_key", "identity_password"):
        if cfg.get(key):
            cfg[key] = "***"
    print(json.dumps(cfg, indent=2, sort_keys=True))


def _run_setup(args) -> None:
    try:
        from hermes_cli.memory_setup import cmd_setup_provider

        cmd_setup_provider("cognee")
    except Exception as exc:
        print(f"Could not launch Hermes memory setup: {exc}")
        print("Run: hermes memory setup cognee")


def _print_install(args) -> None:
    here = Path(__file__).resolve().parents[1]
    name = catalog.catalog_name()
    if name:
        print(f"\nManaged by the Hermes catalog. Update with: hermes plugins update {name}")
        print("Run `hermes memory setup cognee` to make Cognee your memory provider.\n")
        return
    print("\nInstall via the Hermes catalog:")
    print("  hermes plugins install cognee")
    print("  hermes memory setup cognee   # selects Cognee as the memory provider")
    print("  # Updates: hermes plugins update cognee")
    print("\nInstall via pip:")
    print("  pip install cognee-integration-hermes-agent")
    print("  cognee-hermes-install     # copies the plugin into $HERMES_HOME/plugins")
    print("  hermes memory setup cognee")
    print("  # Updates: pip install -U cognee-integration-hermes-agent, then cognee-hermes-install")
    print("\nInstall from a checkout into a Hermes home without an existing Cognee plugin:")
    print("  mkdir -p ~/.hermes/plugins/cognee")
    print(f"  cp -R {here}/. ~/.hermes/plugins/cognee/")
    print("  hermes memory setup cognee")
    print(f"\nLocal mode runtime: hermes cognee install-runtime ({LOCAL_COGNEE_REQUIREMENT}).\n")


def _runtime_status() -> str:
    python = runtime_env.runtime_python()
    if python:
        return f"ready ({runtime_env.LOCAL_COGNEE_REQUIREMENT}, {runtime_env.VENV_DIR})"
    if runtime_env.server_python():
        version = runtime_env.hermes_env_cognee_version()
        return f"cognee {version} installed in Hermes's environment"
    outdated = runtime_env.outdated_requirement() or runtime_env.hermes_env_outdated()
    if outdated:
        where = (
            runtime_env.VENV_DIR if runtime_env.outdated_requirement() else "Hermes's environment"
        )
        return (
            f"outdated ({outdated}, {where}): local mode is off until you "
            f"install {runtime_env.LOCAL_COGNEE_REQUIREMENT} with: hermes cognee install-runtime"
        )
    return "not installed (local mode: hermes cognee install-runtime)"


def _confirm(question: str) -> bool:
    """A yes/no question answered at the terminal; blank is yes, no answer is no."""
    try:
        answer = input(f"{question} (y/n) [y]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in {"", "y", "yes"}


def _install_runtime(args) -> int:
    # Say what will be stopped before stopping it: the server is shared, and
    # another cognee plugin may be using it right now.
    servers = runtime_env.servers_to_stop()
    if servers and not getattr(args, "yes", False):
        print(runtime_env.stop_notice(servers))
        if not sys.stdin.isatty():
            print("Nothing was stopped or installed. Re-run with --yes to go ahead.")
            return 1
        if not _confirm("Continue?"):
            print("Nothing was stopped or installed.")
            return 1
    elif servers:
        print(runtime_env.stop_notice(servers))
    try:
        runtime_env.install_runtime()
    except runtime_env.RuntimeInstallError as exc:
        print(f"Could not install the cognee runtime: {exc}")
        return 1
    print(
        "Start a new hermes session to use local mode. A local cognee server another "
        "cognee plugin started on an older cognee is not stopped for you; until it is "
        "replaced, Hermes reports Cognee as unavailable."
    )
    return 0


def _remove_runtime(args) -> int:
    if runtime_env.remove_runtime():
        print(f"Removed the cognee runtime at {runtime_env.VENV_DIR}.")
        print("A local server it started keeps running until it exits; data in ~/.cognee is kept.")
    else:
        print("No cognee runtime installed.")
    return 0


def cognee_command(args) -> None:
    sub = getattr(args, "cognee_command", None)
    if sub == "setup":
        _run_setup(args)
    elif sub == "install-runtime":
        raise SystemExit(_install_runtime(args))
    elif sub == "remove-runtime":
        raise SystemExit(_remove_runtime(args))
    elif sub == "config":
        _print_config(args)
    elif sub == "install":
        _print_install(args)
    elif sub == "version":
        _print_version(args)
    elif sub == "index-repo":
        raise SystemExit(_run_index_repo(args))
    elif sub == "keys":
        action = getattr(args, "keys_command", None)
        raise SystemExit(_run_keys_revoke(args) if action == "revoke" else _run_keys_list(args))
    elif sub in recap.RECAP_KINDS:
        raise SystemExit(_run_recap(args))
    else:
        _print_status(args)


def register_cli(subparser) -> None:
    """Build the `hermes cognee` command tree."""
    subs = subparser.add_subparsers(dest="cognee_command")
    status = subs.add_parser("status", help="Show Cognee memory status")
    status.add_argument(
        "--check-updates",
        action="store_true",
        help="Check PyPI for pip installs; show Hermes update guidance for catalog installs",
    )
    subs.add_parser("setup", help="Run Hermes memory setup for Cognee")
    subs.add_parser("config", help="Print Cognee plugin config with secrets redacted")
    subs.add_parser("install", help="Print installation commands")
    install_runtime = subs.add_parser(
        "install-runtime",
        help="Install the private cognee environment local mode runs its server from",
        description=(
            "Install or upgrade the private cognee environment local mode runs its server "
            "from. A local cognee server this plugin started is stopped first (SIGTERM, "
            "never SIGKILL), after asking; other cognee plugins using it restart it on their "
            "next request. If a cognee server still answers on the local port, nothing is "
            "changed and the command says which process to stop."
        ),
    )
    install_runtime.add_argument(
        "--yes",
        action="store_true",
        help="Stop a running local server this plugin started without asking (needed "
        "without a terminal)",
    )
    subs.add_parser("remove-runtime", help="Delete the private cognee environment")
    version = subs.add_parser("version", help="Show plugin version and update availability")
    version.add_argument(
        "--check-updates",
        action="store_true",
        help="Check PyPI for pip installs; show Hermes update guidance for catalog installs",
    )
    index = subs.add_parser(
        "index-repo",
        help="Index a repository into a deterministic Cognee code graph (cognee >= 1.5.4)",
    )
    index.add_argument("repo", help="Local path or git URL of the repository")
    index.add_argument(
        "--dataset", default="", help="Dataset name (default: codebase-<repo>-<digest>)"
    )
    index.add_argument(
        "--index-vectors",
        action="store_true",
        help="Also embed code entities for semantic search (needs an embedding provider)",
    )
    index.add_argument(
        "--wait",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Wait up to SECONDS for the indexing pipeline to finish",
    )
    keys = subs.add_parser(
        "keys",
        help="List or revoke the per-machine API keys of your cognee plugin agents",
    )
    keys_subs = keys.add_subparsers(dest="keys_command")
    keys_list = keys_subs.add_parser(
        "list", help="List keys, most recently used first (every plugin agent by default)"
    )
    keys_list.add_argument("--plugin", default="", help="One plugin's agent (e.g. claude-code)")
    keys_list.add_argument("--json", action="store_true")
    keys_revoke = keys_subs.add_parser("revoke", help="Revoke one key by id")
    keys_revoke.add_argument("key_id")
    keys_revoke.add_argument(
        "--plugin", required=True, help="The plugin whose agent holds the key (e.g. claude-code)"
    )
    recap_help = {
        "standup": "What was worked on and left open (default: last 24h)",
        "digest": "Sessions plus the lessons distilled into memory (default: last 7d)",
        "timeline": "A dated chronology of one topic (default: last 30d)",
    }
    for kind in recap.RECAP_KINDS:
        cmd = subs.add_parser(kind, help=recap_help[kind])
        if kind == "timeline":
            cmd.add_argument("topic", help="The topic to trace")
        cmd.add_argument(
            "--since",
            default="",
            help="24h, 7d, 2w, today, yesterday, week, month, all, or a date",
        )
        cmd.add_argument("--json", action="store_true", help="Machine-readable output")
        cmd.add_argument(
            "--all-sessions",
            action="store_true",
            help="Also include other clients' sessions recorded into this dataset",
        )
        cmd.add_argument("--max-sessions", type=int, default=recap.DEFAULT_MAX_SESSIONS)
        cmd.add_argument(
            "--max-learnings",
            type=int,
            default=recap.DEFAULT_MAX_LEARNINGS,
            help="Digest: newest graph learnings to fetch and list (0 = all)",
        )
        cmd.add_argument("--projects", default="", help="Comma-separated repository substrings")
        cmd.add_argument("--dataset", default="", help="Dataset (default: the configured one)")
    subparser.set_defaults(func=cognee_command)
