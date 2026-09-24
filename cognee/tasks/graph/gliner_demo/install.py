"""Install the GLiNER runtime (torch + ``gliner2``) the first time GLiNER is used.

The recommended install for GLiNER users is ``pip install "cognee[gliner]"``. This
module is the fail-safe for users without it, so they can try GLiNER anyway: when a
cognify resolves to the GLiNER extractor and its runtime is missing, it installs the
``gliner`` extra (read from cognee's own metadata, so ``pyproject.toml`` is the only
list) into the running environment:

1. its torch requirement, only when no torch (CPU or GPU build) is installed, from
   the PyTorch CPU index only, so no other package can come from there (PyPI's Linux
   torch pulls several GB of CUDA libraries);
2. the rest of the extra, only when ``gliner2`` is missing, from the default index.

Step 2 runs with every installed distribution pinned to its current version: the
running process may already have imported them, and a package replaced on disk
under a live import breaks it. An install that would need to change one fails.

Every failure raises ``GlinerInstallError`` with the extra as the proposed fix. The
install is blocking; ``ensure_extractor_runtime`` in ``cognee.modules.cognify.config``
runs it in a worker thread and awaits it, so the event loop stays free and the
pipeline starts only once the runtime is importable.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from filelock import FileLock
from packaging.requirements import Requirement

from cognee.exceptions import CogneeConfigurationError
from cognee.shared.logging_utils import get_logger

logger = get_logger("gliner.install")

GLINER_EXTRA = "gliner"
INSTALL_EXTRA_HINT = 'pip install "cognee[gliner]"'
# Seconds between "still installing" status lines while an install step runs.
HEARTBEAT_SECONDS = 15
# Installer output lines worth relaying: what is being fetched and what landed.
_PROGRESS_PREFIXES = (
    "Collecting",
    "Downloading",
    "Installing collected packages",
    "Successfully installed",
    "Resolved",
    "Prepared",
    "Installed",
    "Uninstalled",
)


@dataclass
class InstallOutcome:
    """What an install did, for the caller's log and telemetry (no paths, no output)."""

    installer: str | None = None
    installed: list[str] = field(default_factory=list)
    torch_version: str | None = None
    seconds: int = 0


class GlinerInstallError(CogneeConfigurationError):
    """The GLiNER runtime is missing and could not be installed automatically.

    ``step`` names where it failed (lock, metadata, installer, torch, gliner2,
    verify) and ``installer`` which installer was chosen, for telemetry.
    """

    def __init__(self, message: str, step: str, installer: str | None = None):
        self.step = step
        self.installer = installer
        super().__init__(
            f"Could not install the GLiNER runtime automatically: {message}",
            "GlinerInstallError",
            remediation=(
                f"Install cognee with the GLiNER extra: {INSTALL_EXTRA_HINT} "
                "(or set LLM_API_KEY to extract with an LLM)."
            ),
        )


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def gliner_runtime_installed() -> bool:
    """Whether ``gliner2`` and a torch build (CPU or GPU) are importable."""
    return _installed("gliner2") and _installed("torch")


def gliner_extra() -> tuple[str, list[str]]:
    """The ``gliner`` extra from cognee's installed metadata: (torch requirement, the rest)."""
    torch, rest = None, []
    for line in importlib.metadata.requires("cognee") or []:
        requirement = Requirement(line)
        marker = requirement.marker
        if marker and "extra" in str(marker) and marker.evaluate({"extra": GLINER_EXTRA}):
            requirement.marker = None
            if requirement.name == "torch":
                torch = str(requirement)
            else:
                rest.append(str(requirement))
    if torch is None or not rest:
        raise GlinerInstallError(
            f"cognee's installed metadata has no complete `{GLINER_EXTRA}` extra.",
            "metadata",
        )
    return torch, rest


def installed_pins() -> list[str]:
    """``name==version`` for every distribution installed in this environment."""
    pins = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        if name:
            pins[name.lower()] = f"{name}=={distribution.version}"
    return sorted(pins.values())


def installer_command() -> tuple[str, list[str]]:
    """The installer for this interpreter and its install command: pip when present, else
    uv (uv venvs have no pip)."""
    if _installed("pip"):
        return "pip", [sys.executable, "-m", "pip", "install"]
    uv = shutil.which("uv")
    if uv is not None:
        return "uv", [uv, "pip", "install", "--python", sys.executable]
    raise GlinerInstallError("this environment has neither pip nor uv.", "installer")


def _run(command: list[str], step: str, failed_step: str, installer: str) -> None:
    """Run one install step, relaying its progress lines and a periodic elapsed-time status."""
    started = time.monotonic()
    logger.info("GLiNER runtime: %s ...", step)
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    finished = threading.Event()

    def heartbeat() -> None:
        while not finished.wait(HEARTBEAT_SECONDS):
            logger.info("GLiNER runtime: still %s (%ds elapsed)", step, time.monotonic() - started)

    threading.Thread(target=heartbeat, daemon=True).start()
    tail: list[str] = []
    try:
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            tail = [*tail[-40:], line]
            if line.startswith(_PROGRESS_PREFIXES):
                logger.info("GLiNER runtime: %s", line)
        returncode = process.wait()
    finally:
        finished.set()
    if returncode != 0:
        raise GlinerInstallError(
            f"{step} failed ({' '.join(command)}):\n" + "\n".join(tail), failed_step, installer
        )
    logger.info("GLiNER runtime: done %s in %ds", step, time.monotonic() - started)


def _import_runtime(installer: str) -> None:
    """Import what the extractor loads, here in the install thread.

    Proves the install works in this process (a package can be findable yet fail to
    import), and keeps the first, slow import of freshly installed torch (several
    seconds) off the caller's event loop: later imports hit ``sys.modules``.
    """
    try:
        import torch
        from gliner2 import AutoExtractor
    except Exception as error:
        raise GlinerInstallError(
            f"installed into {sys.prefix}, but this interpreter cannot import it ({error}).",
            "verify",
            installer,
        ) from error


def _install_rest(command: list[str], requirements: list[str], installer: str) -> None:
    """Step 2: the rest of the extra, with every installed distribution pinned."""
    constraints_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as constraints:
            constraints_path = Path(constraints.name)
            constraints.write("\n".join(installed_pins()) + "\n")
        _run(
            [*command, "-c", str(constraints_path), *requirements],
            "installing gliner2 and its tokenizer dependencies (step 2 of 2)",
            "gliner2",
            installer,
        )
    finally:
        if constraints_path is not None:
            constraints_path.unlink(missing_ok=True)


def install_gliner_runtime(
    index_url: str, on_start: Callable[[], None] | None = None
) -> InstallOutcome:
    """Install whichever of torch (CPU build, from ``index_url``) and the extra is missing.

    Blocking. A file lock next to the environment serializes concurrent processes and
    threads; one that waited finds the runtime installed and returns an outcome with
    nothing installed. ``on_start`` is called once this call is the one installing.
    """
    outcome = InstallOutcome()
    lock_path = Path(sys.prefix) / ".cognee-gliner-install.lock"
    try:
        lock = FileLock(str(lock_path))
        lock.acquire()
    except OSError as error:
        raise GlinerInstallError(f"{sys.prefix} is not writable ({error}).", "lock") from error
    try:
        # A process that waited on the lock must see what the holder installed.
        importlib.invalidate_caches()
        if gliner_runtime_installed():
            return outcome
        if on_start is not None:
            on_start()
        torch_requirement, rest = gliner_extra()
        outcome.installer, command = installer_command()
        started = time.monotonic()
        logger.warning(
            "Cognify is extracting the graph with the local GLiNER model, and its runtime "
            "(PyTorch + gliner2) is not installed in %s. Installing it now: one time, about "
            "200 MB to download and 800 MB on disk, CPU-only PyTorch from %s. Set "
            "GLINER_AUTO_INSTALL=false to turn this off, or set LLM_API_KEY to extract "
            "with an LLM instead.",
            sys.prefix,
            index_url,
        )
        if not _installed("torch"):
            _run(
                [*command, "--index-url", index_url, torch_requirement],
                "downloading CPU-only PyTorch (step 1 of 2)",
                "torch",
                outcome.installer,
            )
            outcome.installed.append("torch")
            importlib.invalidate_caches()
        if not _installed("gliner2"):
            _install_rest(command, rest, outcome.installer)
            outcome.installed.append("gliner2")
            importlib.invalidate_caches()
        _import_runtime(outcome.installer)
        outcome.torch_version = importlib.metadata.version("torch")
        outcome.seconds = round(time.monotonic() - started)
        logger.info(
            "GLiNER runtime ready (torch %s) after %ds.", outcome.torch_version, outcome.seconds
        )
        logger.info(
            "Tip: if you keep using GLiNER, install cognee with the extra (%s) so it stays "
            "in your environment instead of being downloaded again.",
            INSTALL_EXTRA_HINT,
        )
        return outcome
    finally:
        lock.release()


if __name__ == "__main__":
    from cognee.modules.cognify.config import get_cognify_config

    install_gliner_runtime(get_cognify_config().gliner_torch_index_url)
