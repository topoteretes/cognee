"""Run the enola binary and parse the snapshot it produces.

enola (https://github.com/enola-labs/enola) is an external Go CLI that
deterministically extracts an architectural graph from a codebase.
`enola --generate <repo>` writes a `.enola/` directory whose contract
artifacts (documented under docs/schema/ upstream, versioned by the receipt's
``format_version``) are `facts.jsonl` (one fact per line), `insights.json`
(explainer findings with evidence), `flows.json` (the ways into the program,
ranked by what each reaches) and `receipt.json` (provenance, counts and
extraction quality). Everything else in the directory is internal.
"""

import asyncio
import hashlib
import json
import os
import platform
import shutil
import sysconfig
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import status

from cognee.exceptions import CogneeConfigurationError, CogneeSystemError
from cognee.shared.logging_utils import get_logger

logger = get_logger("enola")

INSTALL_HINT = (
    "The code graph feature needs the `enola-cli` package, which is included with cognee. "
    "Reinstall cognee to restore it."
)

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

# Where a cluster config is read from when it is piped rather than written:
# enola treats a path that names a file as a config, and reads it once.
_STDIN_CONFIG_PATH = "/dev/stdin"


class EnolaNotInstalledError(CogneeConfigurationError):
    def __init__(
        self,
        message: str = INSTALL_HINT,
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


def _environment_scripts_binary() -> str | None:
    """The enola binary the ``enola-cli`` wheel installs next to this interpreter.

    The wheel puts the binary in the environment's scripts directory. That
    directory is on PATH when the environment is activated, but not when the
    interpreter is run by path (a container entrypoint, a service unit, a
    scheduler), so look there directly before consulting PATH.
    """
    scripts_dir = sysconfig.get_path("scripts")
    if not scripts_dir:
        return None
    suffix = ".exe" if platform.system().lower() == "windows" else ""
    candidate = Path(scripts_dir) / f"enola{suffix}"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return None


def find_enola_binary() -> str:
    """Locate the enola binary: ENOLA_PATH, then this environment's scripts
    directory (where the ``enola-cli`` wheel installs it), then PATH."""
    env_path = os.environ.get("ENOLA_PATH")
    if env_path:
        if os.path.isfile(env_path):
            return env_path
        raise EnolaNotInstalledError(
            message=f"ENOLA_PATH is set to '{env_path}' but no file exists there. {INSTALL_HINT}"
        )

    binary = _environment_scripts_binary() or shutil.which("enola")
    if binary:
        return binary

    raise EnolaNotInstalledError()


async def _run_enola(
    command: list[str],
    cwd: Path,
    timeout: float,
    subject: str,
    stdin_data: bytes | None = None,
) -> None:
    """Run one enola generate command to completion; raise EnolaSnapshotError on failure.

    ``stdin_data`` is fed to the process on its standard input (a cluster
    config passed as ``/dev/stdin``); without it stdin is left unconnected.
    """
    logger.info("Running enola: %s (cwd=%s)", " ".join(command), cwd)

    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(cwd),
        env={**os.environ, **_SUBPROCESS_ENV_OVERRIDES},
        stdin=asyncio.subprocess.PIPE if stdin_data is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        communicating = (
            process.communicate(stdin_data) if stdin_data is not None else process.communicate()
        )
        _stdout, stderr = await asyncio.wait_for(communicating, timeout=timeout)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise EnolaSnapshotError(message=f"enola timed out after {timeout} seconds on {subject}.")

    stderr_text = stderr.decode(errors="replace") if stderr else ""
    if process.returncode != 0:
        # Artifacts already in .enola/ may be from an earlier run; callers
        # must not ingest them (the error propagates before parsing).
        raise EnolaSnapshotError(
            message=(
                f"enola exited with code {process.returncode} on {subject}. "
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


async def run_enola_generate(
    repo_path: str | Path,
    timeout: float = 600.0,
) -> Path:
    """Run `enola --generate` in repo_path and return the snapshot directory.

    Raises EnolaNotInstalledError when no binary is found; nothing is
    downloaded at runtime (the binary ships with cognee's core dependencies).
    """
    binary = find_enola_binary()
    repo_path = Path(repo_path)

    if not repo_path.is_dir():
        raise EnolaSnapshotError(message=f"Repository path '{repo_path}' is not a directory.")

    # The repository is passed explicitly (the documented integration form)
    # AND used as cwd: enola resolves an optional mcp-arch.yaml from the
    # working directory, so this honors a repo-local config while making sure
    # an unrelated one from the caller's cwd can never narrow the run.
    snapshot_dir = repo_path / ".enola"
    await _run_enola([binary, "--generate", str(repo_path)], repo_path, timeout, f"'{repo_path}'")

    if not (snapshot_dir / "facts.jsonl").is_file():
        raise EnolaSnapshotError(
            message=f"enola completed but no facts.jsonl was found in '{snapshot_dir}'."
        )

    return snapshot_dir


def _config_via_stdin() -> bool:
    """Whether enola can read the cluster config from ``/dev/stdin`` here.

    Windows has no ``/dev/stdin``, so the config goes through a temporary
    file there. A function rather than a constant so tests can pick the path
    without patching ``os.name`` (which breaks ``pathlib`` on the running
    interpreter and pytest's own reporting with it).
    """
    return os.name != "nt"


def render_cluster_config(repo_paths: list[Path], output_dir: str) -> str:
    """The enola cluster config for these repositories, as YAML text.

    enola accepts a cluster only as a ``repos:`` config, whose entries must be
    plain path strings. Paths are written absolute, so the config means the
    same thing wherever enola resolves it from, and as JSON strings, which are
    valid YAML scalars and quote any path safely. ``output_dir`` replaces
    ``.enola`` as the snapshot directory; enola joins it to each repository
    path and refuses an absolute one. enola's per-repository
    architecture history (revisions under ``~/.enola/graphs``) is switched
    off: cognee keeps its own record of the snapshot it loaded.
    """
    lines = ["output:", f"  dir: {json.dumps(output_dir)}", "repos:"]
    lines.extend(f"  - {json.dumps(str(path))}" for path in repo_paths)
    lines.extend(["history:", "  enabled: false"])
    return "\n".join(lines) + "\n"


async def run_enola_cluster(
    repo_paths: list[Path],
    output_dir: str,
    timeout: float = 600.0,
) -> Path:
    """Run `enola --generate` over a cluster of repositories; return one snapshot directory.

    enola links the repositories of a cluster config into one graph and
    writes that whole graph to every member's ``output_dir``, so any member's
    directory is the cluster's snapshot. The first member's is returned.

    The config is built from the repositories' paths and handed to enola on
    its standard input (``--generate /dev/stdin``): enola reads the config
    once, at start, so nothing has to be written to disk for it. Windows has
    no ``/dev/stdin``; there the config is a temporary file that is removed
    as soon as enola exits. The working directory is neutral (the system
    temporary directory), so no repository's own mcp-arch.yaml narrows a run
    that covers all of them.
    """
    binary = find_enola_binary()

    for repo_path in repo_paths:
        if not repo_path.is_dir():
            raise EnolaSnapshotError(message=f"Repository path '{repo_path}' is not a directory.")

    config_text = render_cluster_config(repo_paths, output_dir)
    subject = f"the {len(repo_paths)}-repository cluster {[path.name for path in repo_paths]}"
    cwd = Path(tempfile.gettempdir())

    if not _config_via_stdin():
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", prefix="cognee-enola-cluster-", suffix=".yaml", delete=False
        ) as config_file:
            config_file.write(config_text)
        try:
            await _run_enola([binary, "--generate", config_file.name], cwd, timeout, subject)
        finally:
            try:
                os.unlink(config_file.name)
            except OSError:
                pass
    else:
        await _run_enola(
            [binary, "--generate", _STDIN_CONFIG_PATH],
            cwd,
            timeout,
            subject,
            stdin_data=config_text.encode("utf-8"),
        )

    snapshot_dir = repo_paths[0] / output_dir
    if not (snapshot_dir / "facts.jsonl").is_file():
        raise EnolaSnapshotError(
            message=f"enola completed but no facts.jsonl was found in '{snapshot_dir}'."
        )

    return snapshot_dir


@dataclass(frozen=True)
class ParsedSnapshot:
    """A snapshot read once: its directory, facts and receipt.

    extract_code_graph and add_code_graph_edges take it in place of a
    ``snapshot_dir`` so a load that covers several repositories (a dataset's
    cluster) parses ``facts.jsonl`` once, not once per task per member.
    ``edges`` are the typed relation edges already resolved from these facts
    (``build_code_graph_edges``), when the caller resolved them once as well.
    """

    snapshot_dir: Path
    facts: list
    receipt: dict | None
    edges: list | None = None


def parse_enola_snapshot(
    snapshot_dir: str | Path,
) -> tuple[list, dict | None]:
    """Parse facts.jsonl (streamed line by line) and receipt.json from a snapshot dir.

    Blank and corrupt lines are skipped with a warning counter. A missing or
    unparseable receipt.json is not fatal. Findings from insights.json and
    flows from flows.json follow as synthesized facts. Returns (facts, receipt).
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

    repos = {fact["repo"] for fact in facts if isinstance(fact.get("repo"), str) and fact["repo"]}
    insight_facts = _synthesize_insight_facts(snapshot_dir)
    flow_facts = _synthesize_flow_facts(snapshot_dir, repos, receipt)
    if insight_facts or flow_facts:
        facts = facts + insight_facts + flow_facts

    logger.info(
        "Parsed %d fact(s) (%d from insights.json, %d from flows.json) from %s",
        len(facts),
        len(insight_facts),
        len(flow_facts),
        facts_path,
    )
    return facts, receipt


def validate_receipt(
    receipt: dict | None,
    snapshot_dir: str | Path,
    fact_count: int | None = None,
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


def relation_target_id(relation: Any) -> str | None:
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


def _synthesize_flow_facts(snapshot_dir: Path, repos: set[str], receipt: dict | None) -> list:
    """Convert flows.json entry-point flows into fact dicts.

    enola's flows renderer ranks the ways into the program (a served route's
    handler, a ``main``, a symbol a framework invokes) by what the calls from
    each reach. Each flow becomes a synthetic fact of kind "flow" whose
    relations point at its entry (``flow_entry``), its stages
    (``flow_stage``) and the routes it serves (``flow_route``) — through each
    ref's writer-resolved fact ``id`` when present, else by name — so the
    ordinary fact-mapping and edge-resolution paths handle it. The fact keeps
    the flow's own id and takes the entry's file and line.

    A multi-repo run writes every repository's flows into every output
    directory, so only flows of ``repos`` (the repo labels facts.jsonl
    carries) are kept; when facts.jsonl carries no labels, every flow is kept
    and, like the facts, falls back to the snapshot's repository. A missing
    flows.json is not an error (older snapshots lack it, and a configuration
    without the flows renderer writes none).

    Only a flows.json the receipt vouches for is read (_receipt_vouches_for_flows):
    an enola run without the flows renderer leaves an earlier run's file in
    place, and its flows would describe other facts than these.
    """
    flows_path = snapshot_dir / "flows.json"
    if not flows_path.is_file():
        return []
    try:
        content = flows_path.read_bytes()
    except OSError:
        logger.warning("Could not parse flows.json in %s; ignoring it.", snapshot_dir)
        return []
    if not _receipt_vouches_for_flows(receipt, content):
        logger.info(
            "Ignoring flows.json in %s: receipt.json does not list these bytes among this "
            "run's outputs, so another enola run wrote them and their flows may not match "
            "these facts.",
            snapshot_dir,
        )
        return []
    try:
        document = json.loads(content.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        logger.warning("Could not parse flows.json in %s; ignoring it.", snapshot_dir)
        return []
    flows = document.get("flows") if isinstance(document, dict) else None
    if not isinstance(flows, list):
        return []

    facts = []
    for flow in flows:
        if not isinstance(flow, dict):
            continue
        name = flow.get("name")
        if not isinstance(name, str) or not name:
            continue
        repo = flow.get("repo") if isinstance(flow.get("repo"), str) else None
        if repos and repo not in repos:
            continue
        entry = flow.get("entry") if isinstance(flow.get("entry"), dict) else {}
        props = {
            key: flow[key]
            for key in (
                "mechanism",
                "rank",
                "summary",
                "metrics",
                "modules",
                "side",
                "routes",
                "entry",
                "stages",
                "same_reach_as",
                "nested_in",
            )
            if flow.get(key) is not None and (key != "side" or flow[key] is True)
        }
        relations = []
        for relation_kind, refs in (
            ("flow_entry", [entry]),
            ("flow_stage", flow.get("stages")),
            ("flow_route", flow.get("routes")),
        ):
            for ref in refs if isinstance(refs, list) else []:
                if not isinstance(ref, dict):
                    continue
                target = ref.get("name")
                if isinstance(target, str) and target:
                    relation = {"kind": relation_kind, "target": target}
                    if is_enola_id(ref.get("id")):
                        relation["target_id"] = ref["id"]
                    relations.append(relation)
        fact = {"kind": "flow", "name": name, "props": props, "relations": relations}
        if repos:
            # Always one of the facts' own labels here. A label facts.jsonl
            # does not carry would become the fallback repository of every
            # unlabelled fact (the first labelled fact decides it).
            fact["repo"] = repo
        if entry.get("file") is not None:
            fact["file"] = entry["file"]
        if entry.get("line") is not None:
            fact["line"] = entry["line"]
        if is_enola_id(flow.get("id")):
            fact["id"] = flow["id"]
        facts.append(fact)
    _disambiguate_flow_names(facts)
    return facts


def _disambiguate_flow_names(facts: list[dict]) -> None:
    """Give flows that share a name within a repo distinct names, in place.

    A node's identity is (repo, kind, name), and a route flow is named after
    its route: two services serving ``GET /health`` write two flows with
    different ids under one name, which would collapse into one node carrying
    both flows' edges. Each such flow is named after its entry symbol as well,
    then its enola id if that still collides. A flow with a unique name keeps
    it, so its node id does not change; looking up the shared name now
    reports both flows as ambiguous instead of merging them.
    """

    def _groups() -> dict[tuple[str | None, str], list[dict]]:
        groups: dict[tuple[str | None, str], list[dict]] = {}
        for fact in facts:
            groups.setdefault((fact.get("repo"), fact["name"]), []).append(fact)
        return groups

    def _entry_name(fact: dict) -> str | None:
        entry = fact["props"].get("entry")
        name = entry.get("name") if isinstance(entry, dict) else None
        return name if isinstance(name, str) and name else None

    for qualifier in (_entry_name, lambda fact: fact.get("id")):
        for (_repo, name), group in _groups().items():
            if len(group) < 2:
                continue
            for fact in group:
                tag = qualifier(fact)
                if tag and not fact["name"].endswith(f"({tag})"):
                    fact["name"] = f"{name} ({tag})"
    # Still shared (no entry name and no id to tell them apart): number them in
    # flows.json order, which enola ranks deterministically.
    for (_repo, name), group in _groups().items():
        if len(group) > 1:
            for index, fact in enumerate(group, start=1):
                fact["name"] = f"{name} #{index}"


def _receipt_vouches_for_flows(receipt: dict | None, content: bytes) -> bool:
    """Whether receipt.json says this run wrote flows.json, with these bytes.

    enola hashes every artifact it writes into ``output_hashes``, so a file it
    did not write this run is missing there or hashes differently. A receipt
    without output hashes vouches through its ``renderers``. Without a receipt
    there is nothing to check against (historical writers wrote neither a
    receipt nor flows.json); the file is read.
    """
    if not isinstance(receipt, dict):
        return True
    output_hashes = receipt.get("output_hashes")
    if isinstance(output_hashes, dict):
        return output_hashes.get("flows.json") == ("sha256:" + hashlib.sha256(content).hexdigest())
    renderers = receipt.get("renderers")
    return isinstance(renderers, list) and "flows" in renderers


def snapshot_identity(snapshot_dir: str | Path, receipt: dict | None) -> str | None:
    """Stable identity of a snapshot's content, used for incremental skip.

    Prefers receipt.json's snapshot_id — a SHA-256 over enola's byte-stable
    fact serialization, so an unchanged tree produces an unchanged id. Falls
    back to hashing facts.jsonl directly when the receipt is missing. Returns
    None when neither is available; callers must then treat the snapshot as
    changed (load fully, never skip).

    flows.json needs no part in it: only a flows.json the receipt vouches for
    is read, and the snapshot_id also covers the enola version and the
    effective configuration (its renderers included), which decide whether
    and how flows are written.
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


def normalize_relation(relation: dict) -> tuple[str, str] | None:
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
