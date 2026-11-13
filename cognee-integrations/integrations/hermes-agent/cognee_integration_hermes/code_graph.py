"""Code-graph (enola) helpers: repo identity, the recall gate, per-repo state.

Ported from the claude-code/codex plugins' ``_code_graph.py``, minus transport
(the :class:`~.backend.MemoryBackend` owns that) and minus the freshness /
auto-index machinery: Hermes sessions are rarely launched inside a checkout,
so repositories are indexed explicitly — ``hermes cognee index-repo`` — the
way the OpenClaw plugin chose too. Requires a cognee server >= 1.5.4:
1.5.3 opened ``content_type="code"`` on /api/v1/remember and the ``code``
recall scope, and 1.5.4 renamed the repo-spec form field from
``repositories`` to ``raw_data``.

Two responsibilities:

1. **Repo identity + state.** One dataset per indexed repository,
   ``codebase-<repo>-<digest>`` — the path digest is load-bearing (two
   checkouts sharing a basename would otherwise share a graph database, where
   cognee's stale-node sweep lets each re-index delete the other's nodes).
   State lives under ``~/.cognee-plugin/hermes/code-graph/<slug>.json``; a
   repo has a state file once it has been indexed, and only such repos are
   ever matched by the recall lane.

2. **Recall gating** (:func:`auto_code_lane`): the per-prompt recall adds a
   ``code`` scope lane ONLY when (a) the prompt contains an identifier-shaped
   token and (b) the cwd sits inside a repo this plugin has indexed. The lane
   is additive — it never replaces the semantic scopes — and a seed the code
   graph cannot resolve contributes nothing server-side, so misfires are
   cheap by design.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from .config import SHARED_PLUGIN_STATE_DIR

_STATE_DIR = SHARED_PLUGIN_STATE_DIR / "hermes" / "code-graph"

# Mirrors cognee's CodeLoader SUPPORTED_CODE_EXTENSIONS (v1.5.3) — the
# extensions the per-file CODE route claims. Used by the identifier
# extractor's filename pattern.
CODE_EXTENSIONS = frozenset(
    {
        "c", "cc", "cpp", "cs", "cxx", "dart", "fs", "go", "h", "hcl", "hh",
        "hpp", "java", "js", "jsx", "kt", "kts", "php", "proto", "py", "rake",
        "rb", "rs", "scala", "svelte", "swift", "tf", "ts", "tsx", "vb", "vue",
    }
)  # fmt: skip

# CamelCase words that are common prose, product names, or agent vocabulary —
# not symbols worth seeding a code-graph query with. Misfires are cheap
# (seed-not-found returns an empty lane), so this list stays small and only
# covers the words that would fire on nearly every prompt in this ecosystem.
_CAMEL_STOPLIST = frozenset(
    {
        "Claude",
        "ClaudeCode",
        "Codex",
        "Cognee",
        "Hermes",
        "HermesAgent",
        "GitHub",
        "GitLab",
        "JavaScript",
        "TypeScript",
        "PostgreSQL",
        "MongoDB",
        "OpenAI",
        "MacOS",
        "ReadMe",
        "WiFi",
        "OAuth",
        "TODOs",
    }
)

# Ordered by confidence: backticked wins, then file paths, dotted paths,
# snake_case, CamelCase. Each pattern is deliberately conservative — the gate
# exists to keep the code lane OFF conversational prompts, not to catch every
# possible symbol.
_BACKTICK_RE = re.compile(r"`([^`\n]{2,120})`")
_FILEPATH_RE = re.compile(r"\b[\w./-]{1,120}\.(?:%s)\b" % "|".join(sorted(CODE_EXTENSIONS)))
_DOTTED_RE = re.compile(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b")
_SNAKE_RE = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")
_CAMEL_RE = re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+\b")

_IDENTIFIER_CHARS_RE = re.compile(r"^[\w./:-]+$")

# The structured operations cognee's deterministic code route accepts.
CODE_OPERATIONS = (
    "query_facts",
    "explore",
    "traverse",
    "find_path",
    "impact_analysis",
    "delta",
)


def extract_identifiers(prompt: str, limit: int = 2) -> list[str]:
    """Identifier-shaped tokens from a prompt, best first, at most ``limit``.

    An empty list means the syntactic gate did not fire and the code lane
    should be skipped for this prompt.
    """
    if not prompt:
        return []
    found: list[str] = []
    seen: set[str] = set()

    def _add(token: str) -> None:
        token = token.strip().strip(".,:;()[]{}")
        if len(token) < 3 or len(token) > 120:
            return
        if token in _CAMEL_STOPLIST:
            return
        key = token.casefold()
        if key in seen:
            return
        seen.add(key)
        found.append(token)

    for match in _BACKTICK_RE.finditer(prompt):
        inner = match.group(1).strip()
        # Backticks quote commands and prose too; only take single
        # identifier-shaped tokens (no spaces, identifier charset).
        if " " not in inner and _IDENTIFIER_CHARS_RE.match(inner):
            _add(inner)
    for pattern in (_FILEPATH_RE, _DOTTED_RE, _SNAKE_RE, _CAMEL_RE):
        for match in pattern.finditer(prompt):
            token = match.group(0)
            if pattern is _DOTTED_RE:
                # Drop prose artifacts like "e.g" / "i.e": require some meat.
                parts = token.split(".")
                if len(token) < 5 or not any(len(p) >= 3 for p in parts):
                    continue
                # Domain-ish tokens ("example.com") are not symbols.
                if len(parts) == 2 and parts[1].lower() in ("com", "org", "net", "io", "ai", "dev"):
                    continue
            _add(token)

    return found[:limit]


def build_code_query(identifier: str, limit: int = 5) -> dict[str, Any]:
    """The auto-recall code_query: a bounded substring fact lookup.

    query_facts (not explore) on purpose: explore needs a resolvable seed and
    errors on ambiguity, while query_facts degrades to an empty page — the
    right failure mode for a lane that must never disturb the prompt path.
    """
    return {"operation": "query_facts", "name": identifier, "limit": limit}


# ---------------------------------------------------------------------------
# Repo identity + per-repo state
# ---------------------------------------------------------------------------


def is_remote_repo(spec: str) -> bool:
    return isinstance(spec, str) and spec.startswith(("https://", "http://", "git@", "ssh://"))


def canonical_spec(spec: str) -> str:
    """The stable identity of an indexed repository.

    Local paths resolve through symlinks so the same checkout reached by two
    routes is one repository; remote URLs drop a trailing slash and the ``.git``
    suffix so ``…/repo`` and ``…/repo.git`` do not index twice.
    """
    spec = str(spec).strip()
    if is_remote_repo(spec):
        canonical = spec.rstrip("/")
        if canonical.endswith(".git"):
            canonical = canonical[: -len(".git")]
        return canonical
    return os.path.realpath(os.path.expanduser(spec))


def _readable_tail(canonical: str) -> str:
    tail = os.path.basename(canonical.rstrip("/")) or "repo"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", tail).strip("-.") or "repo"


def _repo_slug(root_or_spec: str) -> str:
    canonical = canonical_spec(root_or_spec)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return f"{_readable_tail(canonical)}-{digest}"


def default_code_dataset(spec: str) -> str:
    """A stable, readable, collision-free per-repo dataset name.

    Narrow datasets keep the CODE snapshot cache and the delta pre-read small,
    so one dataset per indexed repository is the default. The basename alone
    cannot be that identity: checkouts routinely share one (``~/work/a/service``
    and ``~/work/b/service``), and two repos landing in one dataset share a
    graph database — where cognee's stale-node sweep, scoped by repo name,
    would let each ingestion delete the other's nodes. The path digest makes
    the name unique per checkout while the readable tail keeps it recognizable
    in a dataset listing.
    """
    canonical = canonical_spec(spec)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]
    # Dots are fine in the repo slug but cognee rejects them in a dataset name
    # (check_dataset_name), so a repo like ``foo.js`` could never be indexed.
    tail = _readable_tail(canonical).lower().replace(".", "-")
    return f"codebase-{tail}-{digest}"


def _state_path(key: str) -> Path:
    return _STATE_DIR / f"{_repo_slug(key)}.json"


def save_repo_state(state: dict[str, Any]) -> None:
    """Persist one repo's index state. ``state`` must carry ``spec``."""
    key = state.get("repo_root") or state.get("spec") or ""
    if not key:
        return
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    _state_path(key).write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def load_repo_states() -> list[dict[str, Any]]:
    """All recorded repo states (invalid files are skipped, never raised)."""
    states: list[dict[str, Any]] = []
    try:
        entries = sorted(_STATE_DIR.glob("*.json"))
    except OSError:
        return states
    for path in entries:
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(state, dict):
            states.append(state)
    return states


def find_indexed_repo(cwd: str) -> dict[str, Any]:
    """The index state whose repo_root contains ``cwd``, or {}.

    Only locally-indexed repos (spec = path) are matched by cwd; a repo
    indexed by URL has no local root to match. Longest matching root wins so
    nested checkouts resolve to the innermost indexed repo.
    """
    real = os.path.realpath(cwd) if cwd else ""
    if not real:
        return {}
    best: dict[str, Any] = {}
    for state in load_repo_states():
        root = str(state.get("repo_root") or "")
        if not root:
            continue
        if real == root or real.startswith(root.rstrip(os.sep) + os.sep):
            if len(root) > len(str(best.get("repo_root") or "")):
                best = state
    return best


def find_repo_state(spec_or_dataset: str) -> dict[str, Any]:
    """The index state matching a repo spec or a code dataset name, or {}."""
    wanted = str(spec_or_dataset or "").strip()
    if not wanted:
        return {}
    canonical = canonical_spec(wanted)
    for state in load_repo_states():
        if state.get("dataset") == wanted or state.get("spec") == canonical:
            return state
    return {}


def auto_code_lane(prompt: str, cwd: str) -> dict[str, Any]:
    """The per-prompt recall gate. Returns {} when the lane must not fire.

    Fires only when the prompt carries an identifier-shaped token AND the
    cwd sits inside a repo this plugin indexed — never on conversational
    prompts, never on repos nobody asked to index.
    """
    identifiers = extract_identifiers(prompt)
    if not identifiers:
        return {}
    state = find_indexed_repo(cwd)
    if not state or not state.get("dataset"):
        return {}
    return {
        "dataset": str(state["dataset"]),
        "identifier": identifiers[0],
        "code_query": build_code_query(identifiers[0]),
    }


def record_index(
    spec: str, dataset: str, *, index_vectors: bool, status: str = "submitted"
) -> dict[str, Any]:
    """Record a successful index submission so the recall lane and re-runs see it."""
    canonical = canonical_spec(spec)
    remote = is_remote_repo(canonical)
    state = {
        "spec": canonical,
        "spec_kind": "url" if remote else "path",
        "repo_root": "" if remote else canonical,
        "dataset": dataset,
        "index_vectors": bool(index_vectors),
        "last_index_at": time.time(),
        "last_status": status,
    }
    save_repo_state(state)
    return state
