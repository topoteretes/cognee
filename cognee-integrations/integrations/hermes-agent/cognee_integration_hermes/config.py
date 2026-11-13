"""Configuration helpers for the Cognee Hermes plugin."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# The other cognee agent plugins (claude-code, codex, openclaw) share one brain:
# one dataset, one store under ~/.cognee, one server on 8011, one minted API key
# under ~/.cognee-plugin. Hermes joins that convention — same names, same paths —
# so memory written in any of them is recalled in all of them.
DEFAULT_DATASET = "agent_sessions"
# Graph recall modes (SDK-1393): "auto" names no dataset and the server picks
# the best-matching one; "pinned" keeps recall on the active dataset.
RECALL_MODE_AUTO = "auto"
RECALL_MODE_PINNED = "pinned"
RECALL_MODES = (RECALL_MODE_AUTO, RECALL_MODE_PINNED)
# Where the configured recall mode came from (load_config's recall_mode_source).
RECALL_MODE_SOURCE_ENV = "env"  # COGNEE_PLUGIN_RECALL_MODE, which wins over cognee.json
RECALL_MODE_SOURCE_FILE = "file"  # recall_mode in cognee.json
RECALL_MODE_SOURCE_DEFAULT = "default"
SHARED_PLUGIN_STATE_DIR = Path.home() / ".cognee-plugin"
SHARED_COGNEE_HOME = Path.home() / ".cognee"
# Port for the local cognee server. 8011 matches the other cognee agent plugins
# and deliberately avoids cognee's own default of 8000, so we never attach to —
# or contend with — a server the user is running themselves.
DEFAULT_LOCAL_PORT = 8011
# How long a boot may take before giving up, matching the other plugins'
# COGNEE_SERVER_BOOT_DEADLINE. A *first* boot runs DB migrations and store
# initialization and can take minutes on a slow machine; giving up early leaves
# memory off for the whole session even though the server finishes booting
# moments later. A genuinely broken spawn never waits this out — the bootstrap
# fails fast once the child is dead and nothing owns the port.
DEFAULT_SERVER_BOOT_TIMEOUT = 600
DEFAULT_IDENTITY_EMAIL = "hermes-agent@cognee.local"
DEFAULT_IDENTITY_PASSWORD = "hermes-agent-plugin"
# The server's *default user*, which the HTTP transport logs in as to mint an
# API key (COGNEE_USER_EMAIL / COGNEE_USER_PASSWORD override). cognee >= 1.6.0
# ships no built-in password: the API server creates the default user only when
# DEFAULT_USER_PASSWORD is set in its environment, so server_bootstrap hands the
# spawned server these same values (an explicit DEFAULT_USER_* export wins) and
# the login keeps working exactly as it did on older servers. Kept here rather
# than in http_backend so the bootstrap can import them without pulling in the
# transport.
DEFAULT_USER_EMAIL = "default_user@example.com"
DEFAULT_USER_PASSWORD = "default_password"
# The cognee release local and embedded mode are written against. It is not a
# dependency of this package: Hermes resolves every plugin into its own shared
# environment, where cognee's pins (huggingface-hub<1, litellm's boto3 floor)
# conflict with Hermes extras and refuse the whole install — including for
# Cloud users, who never import cognee. Remote mode is plain HTTP; local mode
# runs its server from a private venv holding this (runtime_env.py), and
# embedded mode needs it installed into Hermes's environment by hand.
# local-mode-requirements.txt carries the same pin (a test keeps them equal).
# 1.5.4 is the floor (code-graph indexing moved its repo-spec form field to
# raw_data there); 1.6.0 is what the plugin is written against: the password-less
# default user and the only_context prompt contract.
LOCAL_COGNEE_REQUIREMENT = "cognee==1.6.6"
# Recall trimming defaults, the shared reference values (recall_text.py).
DEFAULT_RECALL_PASSAGE_CHARS = 2000
DEFAULT_RECALL_CONTEXT_CHARS = 12000


try:
    from agent.secret_scope import get_secret as _hermes_get_secret
except ImportError:  # older Hermes, or the package imported outside Hermes
    _hermes_get_secret = None


def secret(name: str) -> str:
    """A credential the way Hermes hands it out, ``""`` when unset.

    Hermes's ``get_secret`` honours the active profile, which a bare
    ``os.environ`` read does not when one gateway serves several profiles (and
    the plugin-catalog rules require it). Without that API, read the environment,
    which is where Hermes put the profile's ``.env`` before the API existed.
    """
    if _hermes_get_secret is not None:
        value = _hermes_get_secret(name)
    else:
        value = os.environ.get(name)
    return str(value or "").strip()


def str_to_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def str_to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# Context-length ceilings for the Ollama embedding models we recognize, verified
# against ``ollama show <model>`` ("context length"). Values may sit below a
# model's true context (e.g. sfr-embedding-mistral takes 32k): a low ceiling only
# makes chunks smaller, which is always safe — see ollama_embedding_pins.
_OLLAMA_MODEL_CONTEXT = {
    "all-minilm": 256,
    "nomic-embed-text": 2048,
    "mxbai-embed-large": 512,
    "bge-m3": 8192,
    "embeddinggemma": 2048,
    "qwen3-embedding": 8192,
    "snowflake-arctic-embed": 512,
    "snowflake-arctic-embed2": 8192,
    "sfr-embedding-mistral": 2048,
}
# What a model NOT in this table gets: cognee's own OllamaEmbeddingEngine default,
# small enough for every embedding model Ollama distributes.
_OLLAMA_UNKNOWN_MODEL_CONTEXT = 512

# The HuggingFace tokenizer matching each recognized model, used by cognee to
# count tokens when sizing chunks. Only models whose tokenizer repo is public and
# unambiguous are listed — a wrong tokenizer miscounts tokens and can reintroduce
# the very overflow these pins exist to prevent, so unknown models get
# documentation, never a guess.
_OLLAMA_MODEL_TOKENIZER = {
    "all-minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "nomic-embed-text": "nomic-ai/nomic-embed-text-v1.5",
    "mxbai-embed-large": "mixedbread-ai/mxbai-embed-large-v1",
    "bge-m3": "BAAI/bge-m3",
    "qwen3-embedding": "Qwen/Qwen3-Embedding-0.6B",
    "sfr-embedding-mistral": "Salesforce/SFR-Embedding-Mistral",
}


def _normalize_ollama_model(model: str) -> str:
    """``ollama/nomic-embed-text:latest`` -> ``nomic-embed-text``."""
    name = str(model or "").strip().lower()
    if name.startswith("ollama/"):
        name = name[len("ollama/") :]
    return name.rsplit(":", 1)[0]


def _lookup_ollama_model(table: dict[str, Any], model: str) -> Any:
    """Exact table hit, else the longest key contained in the model name.

    Substring matching absorbs registry namespaces (``avr/sfr-embedding-mistral``)
    and size suffixes (``nomic-embed-text-v1.5``); longest-first keeps
    ``snowflake-arctic-embed2`` from resolving to ``snowflake-arctic-embed``.
    """
    if model in table:
        return table[model]
    for key in sorted(table, key=len, reverse=True):
        if key in model:
            return table[key]
    return None


def ollama_embedding_pins(env: Mapping[str, str]) -> dict[str, str]:
    """Embedding env defaults that keep a local Ollama embedder from silently
    corrupting the search index. Empty unless ``EMBEDDING_PROVIDER=ollama``.

    Why these exist (live-diagnosed on a meeting-notes ingestion): with
    ``EMBEDDING_PROVIDER=ollama`` and nothing else set, cognee sizes cognify
    chunks from ``EMBEDDING_MAX_COMPLETION_TOKENS`` (default 8191) — far above
    any local embedding model's real context — so every substantial document
    overflows. Ollama answers "input exceeds context length"; cognee's
    OllamaEmbeddingEngine then splits the text into overlapping thirds and
    MEAN-POOLS the two vectors, returning success. The pipeline reports
    completed while the vector index quietly fills with lossy embeddings, and
    retrieval degrades with no error anywhere but the server log. Capping the
    token ceiling at the model's true context (the tokenizer pin makes the
    token *count* trustworthy) makes cognee chunk within the model's limits
    instead. A ceiling below the model's context only makes chunks smaller —
    safe — which is why unknown models get a conservative 512 rather than
    nothing.

    Keys the caller's ``env`` already has are omitted (callers also apply these
    with ``setdefault``): an explicit user value always wins.
    """
    provider = str(env.get("EMBEDDING_PROVIDER") or "").strip().lower()
    if provider != "ollama":
        return {}
    model = _normalize_ollama_model(env.get("EMBEDDING_MODEL") or "")
    pins: dict[str, str] = {}
    if "EMBEDDING_MAX_COMPLETION_TOKENS" not in env:
        context = _lookup_ollama_model(_OLLAMA_MODEL_CONTEXT, model)
        pins["EMBEDDING_MAX_COMPLETION_TOKENS"] = str(context or _OLLAMA_UNKNOWN_MODEL_CONTEXT)
    if "HUGGINGFACE_TOKENIZER" not in env:
        tokenizer = _lookup_ollama_model(_OLLAMA_MODEL_TOKENIZER, model)
        if tokenizer:
            pins["HUGGINGFACE_TOKENIZER"] = str(tokenizer)
    return pins


def resolve_hermes_home(hermes_home: str | Path | None = None) -> Path | None:
    if hermes_home:
        return Path(hermes_home).expanduser()
    try:
        from hermes_constants import get_hermes_home

        return Path(get_hermes_home()).expanduser()
    except Exception:
        return None


def configured_dataset_name(value: Any) -> str:
    """A dataset name from the env or config, or the default when it is unset.

    "auto" (any case, surrounding whitespace ignored) is reserved for the
    recall mode (``cognee_switch_dataset action='auto'``), so a configured
    "auto" means the default dataset, never a dataset named "auto". It says
    nothing about the recall mode: that is ``recall_mode``. Same rule as the
    hook-family plugins' ``configured_dataset_name``.
    """
    name = str(value or "").strip()
    if not name or name.lower() == "auto":
        return DEFAULT_DATASET
    return name


def configured_recall_mode(value: Any) -> str:
    """The configured recall mode: "pinned" (any case, surrounding whitespace
    ignored) or "auto" for anything else, unset and unknown values included."""
    mode = str(value or "").strip().lower()
    return RECALL_MODE_PINNED if mode == RECALL_MODE_PINNED else RECALL_MODE_AUTO


def recall_mode_is_set(config: dict[str, Any]) -> bool:
    """True when the user set a recall mode: ``COGNEE_PLUGIN_RECALL_MODE`` or
    ``recall_mode`` in cognee.json (load_config's ``recall_mode_source``)."""
    source = config.get("recall_mode_source")
    if source is None:  # a config dict not built by load_config
        return "recall_mode" in config
    return source in (RECALL_MODE_SOURCE_ENV, RECALL_MODE_SOURCE_FILE)


def effective_recall_mode(config: dict[str, Any], dataset: Any = None) -> str:
    """The configured recall mode in effect.

    A set recall mode decides. With none set, a configured dataset other than
    the default keeps recall pinned to it, as before SDK-1393 (a user who picked
    a dataset to keep agents or projects apart keeps that); the default dataset
    is auto. ``dataset`` overrides ``config["dataset"]`` (the provider passes the
    name it actually uses).
    """
    if recall_mode_is_set(config):
        return configured_recall_mode(config.get("recall_mode"))
    name = configured_dataset_name(config.get("dataset") if dataset is None else dataset)
    return RECALL_MODE_PINNED if name != DEFAULT_DATASET else RECALL_MODE_AUTO


def config_path(hermes_home: str | Path | None = None) -> Path | None:
    home = resolve_hermes_home(hermes_home)
    return home / "cognee.json" if home else None


def resolve_local_roots(config: dict[str, Any]) -> tuple[str, str]:
    """Where cognee should keep its data and system directories.

    Explicit ``COGNEE_DATA_ROOT`` / ``COGNEE_SYSTEM_ROOT`` win. Otherwise both
    default to ``~/.cognee/{data,system}`` — the exact roots the claude-code,
    codex and openclaw plugins pin — so every plugin's server serves the same
    store no matter which of them booted it first.

    The trade-off is deliberate: memory is shared across agents *and* across
    Hermes profiles by default. To isolate a profile, point
    ``COGNEE_DATA_ROOT`` / ``COGNEE_SYSTEM_ROOT`` somewhere else and give it its
    own ``COGNEE_LOCAL_PORT`` — a server belongs to whoever reaches its port
    first, so shared port means shared store regardless of these roots.
    """
    data_root = str(config.get("data_root") or "") or str(SHARED_COGNEE_HOME / "data")
    system_root = str(config.get("system_root") or "") or str(SHARED_COGNEE_HOME / "system")
    return data_root, system_root


def load_config(hermes_home: str | Path | None = None) -> dict[str, Any]:
    """Load plugin config from environment variables and HERMES_HOME/cognee.json."""
    # COGNEE_BASE_URL is the canonical name; COGNEE_SERVICE_URL is a deprecated alias
    # kept for backward compatibility. A set service_url selects remote/cloud mode.
    service_url = secret("COGNEE_BASE_URL") or secret("COGNEE_SERVICE_URL")
    config: dict[str, Any] = {
        "llm_api_key": secret("LLM_API_KEY"),
        "llm_model": os.environ.get("LLM_MODEL", ""),
        "service_url": service_url,
        "api_key": secret("COGNEE_API_KEY"),
        # Connection mode knobs (see provider.initialize / README "Modes").
        # embedded=true runs cognee in-process (single-process/offline only);
        # otherwise local mode ensures a local server on local_port (DB-safe).
        "embedded": str_to_bool(os.environ.get("COGNEE_EMBEDDED"), False),
        # Which transport reaches cognee: "" / "sdk" (default) or "http" for the
        # direct REST client the other cognee plugins use. See backend.build_backend.
        "transport": os.environ.get("COGNEE_TRANSPORT", ""),
        "local_port": str_to_int(os.environ.get("COGNEE_LOCAL_PORT"), DEFAULT_LOCAL_PORT),
        "server_boot_timeout": str_to_int(
            os.environ.get("COGNEE_SERVER_BOOT_TIMEOUT"), DEFAULT_SERVER_BOOT_TIMEOUT
        ),
        # COGNEE_PLUGIN_DATASET is the name the other cognee plugins read;
        # COGNEE_DATASET is this plugin's 0.1.x name, kept as an alias.
        "dataset": os.environ.get("COGNEE_PLUGIN_DATASET")
        or os.environ.get("COGNEE_DATASET", DEFAULT_DATASET),
        # Graph recall mode: "auto" (the server picks the dataset) or "pinned"
        # (recall stays on the active dataset). Same variable as the hook family.
        # Set below, after cognee.json: the exported variable wins over the file.
        "recall_mode": RECALL_MODE_AUTO,
        "top_k": str_to_int(os.environ.get("COGNEE_TOP_K"), 5),
        "auto_route": str_to_bool(os.environ.get("COGNEE_AUTO_ROUTE"), True),
        "improve_on_end": str_to_bool(os.environ.get("COGNEE_IMPROVE_ON_END"), True),
        # Tri-state: "" = auto (background only in server/remote mode, where the
        # server outlives this process; synchronous in embedded). Set to force.
        "improve_background": os.environ.get("COGNEE_IMPROVE_BACKGROUND", ""),
        "session_prefix": os.environ.get("COGNEE_SESSION_PREFIX", "hermes"),
        "data_root": os.environ.get("COGNEE_DATA_ROOT", ""),
        "system_root": os.environ.get("COGNEE_SYSTEM_ROOT", ""),
        "identity_email": os.environ.get("COGNEE_HERMES_USER_EMAIL", DEFAULT_IDENTITY_EMAIL),
        "identity_password": os.environ.get(
            "COGNEE_HERMES_USER_PASSWORD",
            DEFAULT_IDENTITY_PASSWORD,
        ),
        "recall_timeout": str_to_int(os.environ.get("COGNEE_RECALL_TIMEOUT"), 120),
        "write_timeout": str_to_int(os.environ.get("COGNEE_WRITE_TIMEOUT"), 120),
        "improve_timeout": str_to_int(os.environ.get("COGNEE_IMPROVE_TIMEOUT"), 300),
        # Per-prompt memory block: one graph-scope only_context HYBRID_COMPLETION
        # recall carrying the session id (on cognee >= 1.6.0 its text already
        # holds the session history, retrieved context and guidance), plus the
        # code lane when armed. The budget bounds the whole prefetch.
        "recall_budget": str_to_int(os.environ.get("COGNEE_RECALL_BUDGET"), 20),
        # Recall trimming (SDK-904): each recalled passage is capped at
        # recall_passage_chars, and recall_context_chars bounds the whole
        # injection. 0 disables either. Same defaults and variables as openclaw.
        "recall_passage_chars": str_to_int(
            os.environ.get("COGNEE_RECALL_PASSAGE_CHARS"), DEFAULT_RECALL_PASSAGE_CHARS
        ),
        "recall_context_chars": str_to_int(
            os.environ.get("COGNEE_RECALL_CONTEXT_CHARS"), DEFAULT_RECALL_CONTEXT_CHARS
        ),
        # Tokens per chunk for permanent writes (#428); 0 leaves it to the server.
        "chunk_size": str_to_int(os.environ.get("COGNEE_CHUNK_SIZE"), 0),
        # Memory steer: one system-prompt line asserting Cognee as the preferred,
        # authoritative long-term memory (the COGNEE_PREFER_MEMORY counterpart).
        "memory_steer": str_to_bool(os.environ.get("COGNEE_MEMORY_STEER"), True),
        "memory_steer_text": os.environ.get("COGNEE_MEMORY_STEER_TEXT", ""),
        # Per-turn memory-hit visibility in the injected recall block.
        "memory_hits": str_to_bool(os.environ.get("COGNEE_MEMORY_HITS"), True),
        # Agent tools beyond the recall/remember/forget trio.
        "dataset_switch_tool": str_to_bool(os.environ.get("COGNEE_DATASET_SWITCH_TOOL"), True),
        "code_search_tool": str_to_bool(os.environ.get("COGNEE_CODE_SEARCH_TOOL"), True),
        "recap_tool": str_to_bool(os.environ.get("COGNEE_RECAP_TOOL"), True),
        # The identifier-gated code recall lane, plus extra always-on code
        # datasets (comma-separated) for repos indexed from another machine.
        "code_graph_recall": str_to_bool(os.environ.get("COGNEE_CODE_GRAPH_RECALL"), True),
        "code_datasets": os.environ.get("COGNEE_CODE_DATASETS", ""),
        # PyPI update check (CLI only, never on the session path).
        "update_check": str_to_bool(os.environ.get("COGNEE_UPDATE_CHECK"), True),
        "update_check_interval": str_to_int(os.environ.get("COGNEE_UPDATE_CHECK_INTERVAL"), 3600),
    }

    path = config_path(hermes_home)
    file_recall_mode = None
    if path and path.exists():
        try:
            file_config = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(file_config, dict):
                config.update(
                    {key: value for key, value in file_config.items() if value is not None}
                )
                file_recall_mode = file_config.get("recall_mode")
        except Exception:
            pass

    # "auto" is reserved for the recall mode, never a dataset: saves go to the
    # default (configured_dataset_name). It does not set the mode. A blank name
    # is left as it is (see the tests).
    if str(config.get("dataset") or "").strip().lower() == "auto":
        config["dataset"] = DEFAULT_DATASET
    # The recall mode is the one setting an exported variable overrides
    # cognee.json for: COGNEE_PLUGIN_RECALL_MODE is the hook family's setting of
    # the same name, and the setup wizard always writes recall_mode, so the file
    # would otherwise silence it. recall_mode_source says which one applied.
    env_recall_mode = os.environ.get("COGNEE_PLUGIN_RECALL_MODE", "").strip()
    if env_recall_mode:
        config["recall_mode"] = env_recall_mode
        config["recall_mode_source"] = RECALL_MODE_SOURCE_ENV
    elif file_recall_mode is not None:
        config["recall_mode_source"] = RECALL_MODE_SOURCE_FILE
    else:
        config["recall_mode_source"] = RECALL_MODE_SOURCE_DEFAULT
    config["recall_mode"] = configured_recall_mode(config.get("recall_mode"))
    config["top_k"] = max(1, str_to_int(config.get("top_k"), 5))
    config["recall_timeout"] = max(1, str_to_int(config.get("recall_timeout"), 120))
    config["write_timeout"] = max(1, str_to_int(config.get("write_timeout"), 120))
    config["improve_timeout"] = max(1, str_to_int(config.get("improve_timeout"), 300))
    config["local_port"] = min(
        65535, max(1, str_to_int(config.get("local_port"), DEFAULT_LOCAL_PORT))
    )
    config["server_boot_timeout"] = max(
        1, str_to_int(config.get("server_boot_timeout"), DEFAULT_SERVER_BOOT_TIMEOUT)
    )
    config["auto_route"] = str_to_bool(config.get("auto_route"), True)
    config["improve_on_end"] = str_to_bool(config.get("improve_on_end"), True)
    config["embedded"] = str_to_bool(config.get("embedded"), False)
    config["recall_budget"] = max(1, str_to_int(config.get("recall_budget"), 20))
    for key, default in (
        ("recall_passage_chars", DEFAULT_RECALL_PASSAGE_CHARS),
        ("recall_context_chars", DEFAULT_RECALL_CONTEXT_CHARS),
    ):
        value = str_to_int(config.get(key), default)
        config[key] = value if value >= 0 else default
    config["chunk_size"] = max(0, str_to_int(config.get("chunk_size"), 0))
    config["memory_steer"] = str_to_bool(config.get("memory_steer"), True)
    config["memory_hits"] = str_to_bool(config.get("memory_hits"), True)
    config["dataset_switch_tool"] = str_to_bool(config.get("dataset_switch_tool"), True)
    config["code_search_tool"] = str_to_bool(config.get("code_search_tool"), True)
    config["recap_tool"] = str_to_bool(config.get("recap_tool"), True)
    config["code_graph_recall"] = str_to_bool(config.get("code_graph_recall"), True)
    config["update_check"] = str_to_bool(config.get("update_check"), True)
    config["update_check_interval"] = max(60, str_to_int(config.get("update_check_interval"), 3600))
    return config


def save_config(values: dict[str, Any], hermes_home: str | Path) -> Path:
    """Merge non-secret values into HERMES_HOME/cognee.json."""
    path = config_path(hermes_home)
    if path is None:
        raise RuntimeError("Could not resolve HERMES_HOME.")
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                existing = loaded
        except Exception:
            existing = {}

    existing.update({key: value for key, value in values.items() if value is not None})
    path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def save_env_vars(values: dict[str, str], hermes_home: str | Path | None = None) -> None:
    """Store setup's values in the profile's ``.env`` through Hermes's own writer.

    The plugin never opens a ``.env`` itself (plugin-catalog rule 11): Hermes's
    ``save_env_value`` validates the name, strips CR/LF and writes 0600
    atomically. A non-default profile's home is applied through Hermes's
    context-local override, the way Hermes's own memory setup does it. An empty
    value clears a key (how the deprecated ``COGNEE_SERVICE_URL`` is retired).
    """
    if not values:
        return
    from hermes_cli.config import save_env_value

    try:
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    except ImportError:  # older Hermes: save_env_value writes the active home
        set_hermes_home_override = reset_hermes_home_override = None
    token = None
    if hermes_home is not None and set_hermes_home_override is not None:
        token = set_hermes_home_override(str(hermes_home))
    try:
        for key, value in values.items():
            try:
                save_env_value(key, value)
            except ValueError as exc:  # a name Hermes refuses to write
                print(f"  Skipping {key}: {exc}")
    finally:
        if token is not None:
            reset_hermes_home_override(token)
