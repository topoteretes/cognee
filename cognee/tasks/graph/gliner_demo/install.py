"""Install the GLiNER runtime (torch + ``gliner2``) the first time GLiNER is used.

PyPI's Linux torch wheels bundle the CUDA libraries (several GB), so torch is not a
dependency of ``cognee`` itself. When a cognify resolves to the GLiNER extractor and
its runtime is missing, this module installs it into the running environment:

1. torch, only when no torch (CPU or GPU build) is installed, from the PyTorch CPU
   index only, so no other package can come from there;
2. the ``gliner`` extra's requirements (read from cognee's own metadata), only when
   ``gliner2`` is missing, from the default index.

Step 2 runs with every installed distribution pinned to its current version: the
running process may already have imported them, and a package replaced on disk
under a live import breaks it. An install that would need to change one fails with
the resolver's message instead.

Users who manage torch themselves install ``cognee[gliner]`` (torch from PyPI: the
CUDA build on Linux) or run ``python -m cognee.tasks.graph.gliner_demo.install`` at
build time (the Docker image does) and may set ``GLINER_AUTO_INSTALL=false``.
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
from pathlib import Path

from filelock import FileLock
from packaging.requirements import Requirement

from cognee.shared.logging_utils import get_logger

logger = get_logger("gliner.install")

TORCH_REQUIREMENT = "torch>=2.1,<3"
DEFAULT_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cpu"
GLINER_EXTRA = "gliner"
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


class GlinerInstallError(RuntimeError):
    """The GLiNER runtime is missing and could not be installed."""


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def gliner_runtime_installed() -> bool:
    """Whether ``gliner2`` and a torch build (CPU or GPU) are importable."""
    return _installed("gliner2") and _installed("torch")


def gliner_requirements() -> list[str]:
    """The ``gliner`` extra's requirements, as declared in cognee's installed metadata."""
    requirements = []
    for line in importlib.metadata.requires("cognee") or []:
        requirement = Requirement(line)
        marker = requirement.marker
        if marker and "extra" in str(marker) and marker.evaluate({"extra": GLINER_EXTRA}):
            requirement.marker = None
            requirements.append(str(requirement))
    if not requirements:
        raise GlinerInstallError(
            f"cognee's installed metadata declares no `{GLINER_EXTRA}` extra; "
            "reinstall cognee so its metadata is complete."
        )
    return requirements


def installed_pins() -> list[str]:
    """``name==version`` for every distribution installed in this environment."""
    pins = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        if name:
            pins[name.lower()] = f"{name}=={distribution.version}"
    return sorted(pins.values())


def installer_command() -> list[str]:
    """The install command for this interpreter: pip when present, else uv (uv venvs have no pip)."""
    if _installed("pip"):
        return [sys.executable, "-m", "pip", "install"]
    uv = shutil.which("uv")
    if uv is not None:
        return [uv, "pip", "install", "--python", sys.executable]
    raise GlinerInstallError(
        "Cannot install the GLiNER runtime: this environment has neither pip nor uv. "
        f"Install it yourself: pip install torch --index-url {DEFAULT_TORCH_INDEX_URL} "
        '&& pip install "cognee[gliner]"'
    )


def _run(command: list[str], step: str) -> None:
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
            f"Installing the GLiNER runtime failed while {step} ({' '.join(command)}):\n"
            + "\n".join(tail)
        )
    logger.info("GLiNER runtime: done %s in %ds", step, time.monotonic() - started)


def install_gliner_runtime(index_url: str = DEFAULT_TORCH_INDEX_URL) -> None:
    """Install whichever of torch (CPU build) and the ``gliner`` extra is missing.

    A file lock next to the environment serializes concurrent processes; one that
    waited finds the runtime installed and returns.
    """
    with FileLock(str(Path(sys.prefix) / ".cognee-gliner-install.lock")):
        # A process that waited on the lock must see what the holder installed.
        importlib.invalidate_caches()
        if gliner_runtime_installed():
            return
        command = installer_command()
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
                [*command, "--index-url", index_url, TORCH_REQUIREMENT],
                "downloading CPU-only PyTorch (step 1 of 2)",
            )
            importlib.invalidate_caches()
        if not _installed("gliner2"):
            with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as constraints:
                constraints.write("\n".join(installed_pins()) + "\n")
            try:
                _run(
                    [*command, "-c", constraints.name, *gliner_requirements()],
                    "installing gliner2 and its tokenizer dependencies (step 2 of 2)",
                )
            finally:
                Path(constraints.name).unlink()
            importlib.invalidate_caches()
        if not gliner_runtime_installed():
            raise GlinerInstallError(
                f"Installed the GLiNER runtime into {sys.prefix}, but this interpreter "
                "still cannot import gliner2 and torch."
            )
        logger.info(
            "GLiNER runtime ready (torch %s) after %ds.",
            importlib.metadata.version("torch"),
            time.monotonic() - started,
        )


if __name__ == "__main__":
    from cognee.modules.cognify.config import get_cognify_config

    install_gliner_runtime(get_cognify_config().gliner_torch_index_url)
