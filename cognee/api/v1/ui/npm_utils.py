import platform
import subprocess
from pathlib import Path
from typing import List

from cognee.shared.logging_utils import get_logger

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
        if result.returncode != 0 and (Path.home() / ".nvm" / "nvm.sh").exists():
            nvm_cmd = f"source ~/.nvm/nvm.sh 2>/dev/null && {' '.join(cmd)}"
            result = subprocess.run(
                ["bash", "-c", nvm_cmd],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        return result
