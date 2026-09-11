"""Run the enola binary and parse the snapshot it produces.

enola (https://github.com/enola-labs/enola) is an external Go CLI that
deterministically extracts an architectural graph from a codebase.
`enola --generate <repo>` writes a `.enola/` directory whose contract
artifacts (documented under docs/schema/ upstream, versioned by the receipt's
``format_version``) are `facts.jsonl` (one fact per line), `insights.json`
(explainer findings with evidence) and `receipt.json` (provenance, counts and
extraction quality). Everything else in the directory is internal.
"""

import asyncio
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Optional, Tuple, Union

from fastapi import status

from cognee.exceptions import CogneeConfigurationError, CogneeSystemError
from cognee.shared.logging_utils import get_logger

logger = get_logger("enola")

ENOLA_INSTALL_URL = "https://github.com/enola-labs/enola#installation"

# Snapshot artifact format generations this reader understands (receipt.json
# ``format_version``, written since enola 0.4.10). Additive vocabulary — new
# kinds, relation kinds, props — never bumps it; a bump means renamed fields or
# changed identity semantics, which this reader must not guess its way past.
# A receipt without the field is a historical writer and reads as version 1.
SUPPORTED_FORMAT_VERSIONS = frozenset({1})

# Fact identity as written by enola >= 0.4.10: sha256(repo, kind, name, file)
# truncated to 128 bits, 32 lowercase hex characters.
_ENOLA_ID_LENGTH = 32
_HEX_DIGITS = frozenset("0123456789abcdef")

# The documented relation shape is {kind, target, target_id?}. Historical
# writers were undocumented, so the plausible spellings for the relation type
# and the target name are still probed; entries that normalize to neither are
# skipped.
_RELATION_TYPE_KEYS = ("kind", "type", "relation", "rel")
_RELATION_TARGET_KEYS = ("target", "name", "to", "target_name")

# Environment for the generate subprocess. Cognee pins the enola release it
# runs, so enola's own release check (a GitHub API call, cached 12h under
# ~/.enola) is pure noise here — and the run must stay offline-safe. Prompts
# and terminal hints assume an interactive shell.
_SUBPROCESS_ENV_OVERRIDES = {"ENOLA_NO_UPDATE_CHECK": "1", "ENOLA_NO_PROMPTS": "1"}


class EnolaNotInstalledError(CogneeConfigurationError):
    def __init__(
        self,
        message: str = (
            "The enola binary was not found. Install it from "
            f"{ENOLA_INSTALL_URL} and make sure it is on PATH, "
            "or point the ENOLA_PATH environment variable at the binary."
        ),
        name: str = "EnolaNotInstalledError",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ):
        super().__init__(message, name, status_code)


class EnolaSnapshotError(CogneeSystemError):
    def __init__(
        self,
        message: str = "enola failed to generate a snapshot.",
        name: str = "EnolaSnapshotError",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ):
        super().__init__(message, name, status_code)


def find_enola_binary() -> str:
    """Locate the enola binary via ENOLA_PATH, falling back to PATH lookup."""
    env_path = os.environ.get("ENOLA_PATH")
    if env_path:
        if os.path.isfile(env_path):
            return env_path
        raise EnolaNotInstalledError(
            message=(
                f"ENOLA_PATH is set to '{env_path}' but no file exists there. "
                f"Install enola from {ENOLA_INSTALL_URL} or fix ENOLA_PATH."
            )
        )

    binary = shutil.which("enola")
    if binary:
        return binary

    raise EnolaNotInstalledError()


async def run_enola_generate(
    repo_path: Union[str, Path],
    timeout: float = 600.0,
) -> Path:
    """Run `enola --generate` in repo_path and return the snapshot directory.

    When the binary is missing (and ENOLA_PATH is not explicitly set), the
    pinned release is downloaded and installed automatically; see
    install_enola.py. Disable with ENOLA_AUTO_INSTALL=false.
    """
    binary = None
    try:
        binary = find_enola_binary()
    except EnolaNotInstalledError:
        from cognee.tasks.code_graph.install_enola import auto_install_enabled, install_enola

        if os.environ.get("ENOLA_PATH") or not auto_install_enabled():
            raise
        binary = await asyncio.to_thread(install_enola)
    repo_path = Path(repo_path)

    if not repo_path.is_dir():
        raise EnolaSnapshotError(message=f"Repository path '{repo_path}' is not a directory.")

    # The repository is passed explicitly (the documented integration form)
    # AND used as cwd: enola resolves an optional mcp-arch.yaml from the
    # working directory, so this honors a repo-local config while making sure
    # an unrelated one from the caller's cwd can never narrow the run.
    command = [binary, "--generate", str(repo_path)]
    snapshot_dir = repo_path / ".enola"

    logger.info("Running enola: %s (cwd=%s)", " ".join(command), repo_path)

    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(repo_path),
        env={**os.environ, **_SUBPROCESS_ENV_OVERRIDES},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        _stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise EnolaSnapshotError(
            message=f"enola timed out after {timeout} seconds on '{repo_path}'."
        )

    stderr_text = stderr.decode(errors="replace") if stderr else ""
    if process.returncode != 0:
        # Artifacts already in .enola/ may be from an earlier run; callers
        # must not ingest them (the error propagates before parsing).
        raise EnolaSnapshotError(
            message=(
                f"enola exited with code {process.returncode} on '{repo_path}'. "
                f"stderr tail: {stderr_text[-2000:]}"
            )
        )

    # enola reports the configuration it resolved on stderr ("enola: using
    # config ..." / "enola: no mcp-arch.yaml in ..., using built-in defaults").
    # A config decides which extractors run and which paths are ignored, so
    # the line belongs in the ingestion log next to the snapshot it shaped.
    for line in stderr_text.splitlines():
        if line.startswith("enola:"):
            logger.info("%s", line.strip())

    if not (snapshot_dir / "facts.jsonl").is_file():
        raise EnolaSnapshotError(
            message=f"enola completed but no facts.jsonl was found in '{snapshot_dir}'."
        )

    return snapshot_dir


def parse_enola_snapshot(
    snapshot_dir: Union[str, Path],
) -> Tuple[list, Optional[dict]]:
    """Parse facts.jsonl (streamed line by line) and receipt.json from a snapshot dir.

    Blank and corrupt lines are skipped with a warning counter. A missing or
    unparseable receipt.json is not fatal. Returns (facts, receipt).
    """
    snapshot_dir = Path(snapshot_dir)
    facts_path = snapshot_dir / "facts.jsonl"

    if not facts_path.is_file():
        raise EnolaSnapshotError(message=f"No facts.jsonl found in '{snapshot_dir}'.")

    facts = []
    corrupt_lines = 0

    with open(facts_path, "r", encoding="utf-8") as facts_file:
        for line_number, line in enumerate(facts_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                fact = json.loads(line)
            except json.JSONDecodeError:
                corrupt_lines += 1
                logger.warning("Skipping corrupt JSON on line %d of %s", line_number, facts_path)
                continue
            if not isinstance(fact, dict):
                corrupt_lines += 1
                logger.warning("Skipping non-object fact on line %d of %s", line_number, facts_path)
                continue
            facts.append(fact)

    if corrupt_lines:
        logger.warning("Skipped %d corrupt line(s) in %s", corrupt_lines, facts_path)

    receipt = None
    receipt_path = snapshot_dir / "receipt.json"
    if receipt_path.is_file():
        try:
            with open(receipt_path, "r", encoding="utf-8") as receipt_file:
                loaded = json.load(receipt_file)
            receipt = loaded if isinstance(loaded, dict) else None
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not parse receipt.json in %s; ignoring it.", snapshot_dir)

    validate_receipt(receipt, snapshot_dir, fact_count=len(facts))

    insight_facts = _synthesize_insight_facts(snapshot_dir)
    if insight_facts:
        facts = facts + insight_facts

    logger.info(
        "Parsed %d fact(s) (%d from insights.json) from %s",
        len(facts),
        len(insight_facts),
        facts_path,
    )
    return facts, receipt


def validate_receipt(
    receipt: Optional[dict],
    snapshot_dir: Union[str, Path],
    fact_count: Optional[int] = None,
) -> None:
    """Reject unsupported artifact formats and surface extraction-quality signals.

    Follows enola's integration contract: an unsupported ``format_version``
    (including ``0``, which is "unknown") is a hard error, a missing one is a
    historical writer and reads as version 1. The receipt's ``fact_count`` is
    checked against what facts.jsonl yielded, and the ``quality`` block's
    parse errors and skip census are logged — they are signals to surface,
    not rejection thresholds. A missing receipt validates trivially.
    """
    if not isinstance(receipt, dict):
        return

    format_version = receipt.get("format_version")
    if format_version is not None and (
        isinstance(format_version, bool)
        or not isinstance(format_version, int)
        or format_version not in SUPPORTED_FORMAT_VERSIONS
    ):
        raise EnolaSnapshotError(
            message=(
                f"Unsupported enola snapshot format_version {format_version!r} in "
                f"'{snapshot_dir}' (enola {receipt.get('enola_version', '?')}); this cognee "
                f"reads format version(s) {sorted(SUPPORTED_FORMAT_VERSIONS)}. Upgrade cognee, "
                "or pin an enola release that writes a supported format."
            )
        )

    declared_count = receipt.get("fact_count")
    if (
        fact_count is not None
        and isinstance(declared_count, int)
        and not isinstance(declared_count, bool)
        and declared_count != fact_count
    ):
        logger.warning(
            "receipt.json declares %d fact(s) but facts.jsonl in %s yielded %d; "
            "the snapshot may be truncated or from a different run.",
            declared_count,
            snapshot_dir,
            fact_count,
        )

    quality = receipt.get("quality")
    if not isinstance(quality, dict):
        return
    parse_errors = quality.get("parse_errors")
    if isinstance(parse_errors, int) and parse_errors > 0:
        logger.warning(
            "enola reported %d parse error(s) for %s; the code graph may be incomplete. Sample: %s",
            parse_errors,
            snapshot_dir,
            quality.get("parse_error_sample"),
        )
    files_seen = quality.get("files_seen")
    files_parsed = quality.get("files_parsed")
    if isinstance(files_seen, int) and isinstance(files_parsed, int) and files_parsed < files_seen:
        census = quality.get("census") if isinstance(quality.get("census"), dict) else {}
        logger.info(
            "enola parsed %d of %d source file(s) in %s (top skip causes: %s).",
            files_parsed,
            files_seen,
            snapshot_dir,
            census.get("top_skip_causes") or quality.get("skipped_sample") or "n/a",
        )


def is_enola_id(value: Any) -> bool:
    """Whether value is a writer fact identity (32 lowercase hex chars, enola >= 0.4.10)."""
    return (
        isinstance(value, str)
        and len(value) == _ENOLA_ID_LENGTH
        and all(character in _HEX_DIGITS for character in value)
    )


def relation_target_id(relation: Any) -> Optional[str]:
    """The writer-resolved target identity of a relation (``target_id``), or None.

    enola emits it only when the target resolves unambiguously (same-repo
    facts first, then snapshot-wide); when absent the readable ``target`` name
    is all there is, and the consumer must not pick an arbitrary match.
    """
    if not isinstance(relation, dict):
        return None
    value = relation.get("target_id")
    return value if is_enola_id(value) else None


def _synthesize_insight_facts(snapshot_dir: Path) -> list:
    """Convert insights.json explainer findings into fact dicts.

    enola's explainers (cycles, layers, hotspots, god-class, dead-methods,
    unused-routes, ...) write architecture findings to insights.json. Each
    becomes a synthetic fact of kind "insight" whose relations point at the
    evidence facts it cites — through the evidence's writer-resolved
    ``fact_id`` when present, else by name — so the ordinary fact-mapping and
    edge-resolution paths handle it. The machine-readable ``metrics`` block
    and the ``informational`` flag ride along in props. A missing or
    unparseable insights.json is not an error (0.1.x snapshots lack it).
    """
    insights_path = snapshot_dir / "insights.json"
    if not insights_path.is_file():
        return []
    try:
        with open(insights_path, "r", encoding="utf-8") as insights_file:
            insights = json.load(insights_file)
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not parse insights.json in %s; ignoring it.", snapshot_dir)
        return []
    if not isinstance(insights, list):
        return []

    facts = []
    for insight in insights:
        if not isinstance(insight, dict):
            continue
        title = insight.get("title")
        if not isinstance(title, str) or not title:
            continue
        props = {
            key: insight[key]
            for key in (
                "source",
                "confidence",
                "description",
                "suggested_actions",
                "metrics",
                "informational",
            )
            if insight.get(key) is not None
        }
        relations = []
        evidence = insight.get("evidence")
        for entry in evidence if isinstance(evidence, list) else []:
            if not isinstance(entry, dict):
                continue
            # Same precedence the writer uses to resolve fact_id: symbol, then
            # fact; a bare file citation falls back to the file's own fact.
            target = entry.get("symbol") or entry.get("fact") or entry.get("file")
            if isinstance(target, str) and target:
                relation = {"kind": "evidences", "target": target}
                fact_id = entry.get("fact_id")
                if is_enola_id(fact_id):
                    relation["target_id"] = fact_id
                relations.append(relation)
        facts.append(
            {
                "kind": "insight",
                "name": title,
                "props": props,
                "relations": relations,
            }
        )
    return facts


def snapshot_identity(snapshot_dir: Union[str, Path], receipt: Optional[dict]) -> Optional[str]:
    """Stable identity of a snapshot's content, used for incremental skip.

    Prefers receipt.json's snapshot_id — a SHA-256 over enola's byte-stable
    fact serialization, so an unchanged tree produces an unchanged id. Falls
    back to hashing facts.jsonl directly when the receipt is missing. Returns
    None when neither is available; callers must then treat the snapshot as
    changed (load fully, never skip).
    """
    if receipt:
        snapshot_id = receipt.get("snapshot_id")
        if isinstance(snapshot_id, str) and snapshot_id:
            return snapshot_id
    facts_path = Path(snapshot_dir) / "facts.jsonl"
    try:
        return "sha256:" + hashlib.sha256(facts_path.read_bytes()).hexdigest()
    except OSError:
        return None


def normalize_relation(relation: dict) -> Optional[Tuple[str, str]]:
    """Extract (relation_type, target_name) from a relation object, or None.

    Probes the alternate key spellings enola may use; returns None when either
    the relation type or the target name cannot be found.
    """
    if not isinstance(relation, dict):
        return None

    relation_type = None
    for key in _RELATION_TYPE_KEYS:
        value = relation.get(key)
        if isinstance(value, str) and value:
            relation_type = value
            break

    target_name = None
    for key in _RELATION_TARGET_KEYS:
        value = relation.get(key)
        if isinstance(value, str) and value:
            target_name = value
            break

    if relation_type is None or target_name is None:
        return None

    return relation_type, target_name
