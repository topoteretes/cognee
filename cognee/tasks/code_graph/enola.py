"""Run the enola binary and parse the snapshot it produces.

enola (https://github.com/enola-labs/enola) is an external Go CLI that
deterministically extracts an architectural graph from a codebase.
`enola --generate` writes a `.enola/` directory containing `facts.jsonl`
(one graph node per line) and `receipt.json` (provenance).
"""

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple, Union

from fastapi import status

from cognee.exceptions import CogneeConfigurationError, CogneeSystemError
from cognee.shared.logging_utils import get_logger

logger = get_logger("enola")

ENOLA_INSTALL_URL = "https://github.com/enola-labs/enola#installation"

# The exact JSON field names inside a relation object are not documented
# (Go struct: Relation), so we probe the plausible spellings for the relation
# type and the target node name and skip entries we cannot normalize.
_RELATION_TYPE_KEYS = ("type", "kind", "relation", "rel")
_RELATION_TARGET_KEYS = ("target", "name", "to", "target_name")


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

    command = [binary, "--generate"]
    snapshot_dir = repo_path / ".enola"

    logger.info("Running enola: %s (cwd=%s)", " ".join(command), repo_path)

    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(repo_path),
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

    if process.returncode != 0:
        stderr_tail = stderr.decode(errors="replace")[-2000:] if stderr else ""
        raise EnolaSnapshotError(
            message=(
                f"enola exited with code {process.returncode} on '{repo_path}'. "
                f"stderr tail: {stderr_tail}"
            )
        )

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

    logger.info("Parsed %d fact(s) from %s", len(facts), facts_path)
    return facts, receipt


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
