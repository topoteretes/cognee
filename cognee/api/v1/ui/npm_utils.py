import platform
import subprocess
from pathlib import Path
from typing import List

from cognee.shared.logging_utils import get_logger
from .node_setup import get_nvm_sh_path

logger = get_logger()


def run_npm_command(cmd: List[str], cwd: Path, timeout: int = 300) -> subprocess.CompletedProcess:
    """
    Run an npm command, ensuring nvm is sourced if needed (Unix-like systems only).
    Returns the CompletedProcess result.
    """
    if platform.system() == "Windows":
        # On Windows, use shell=True for npm commands
        return subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            shell=True,
        )
    else:
        # On Unix-like systems, try direct command first
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        # If it fails and nvm might be installed, try with nvm sourced
        if result.returncode != 0:
            nvm_path = get_nvm_sh_path()
            if nvm_path.exists():
                nvm_cmd = f"source {nvm_path} && {' '.join(cmd)}"
                result = subprocess.run(
                    ["bash", "-c", nvm_cmd],
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                if result.returncode != 0 and result.stderr:
                    logger.debug(f"npm command failed with nvm: {result.stderr.strip()}")
        return result
