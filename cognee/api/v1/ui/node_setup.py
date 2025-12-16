import os
import platform
import subprocess
import tempfile
from pathlib import Path

import requests

from cognee.shared.logging_utils import get_logger

logger = get_logger()


def get_nvm_dir() -> Path:
    """
    Get the nvm directory path following standard nvm installation logic.
    Uses XDG_CONFIG_HOME if set, otherwise falls back to ~/.nvm.
    """
    xdg_config_home = os.environ.get("XDG_CONFIG_HOME")
    if xdg_config_home:
        return Path(xdg_config_home) / "nvm"
    return Path.home() / ".nvm"


def get_nvm_sh_path() -> Path:
    """
    Get the path to nvm.sh following standard nvm installation logic.
    """
    return get_nvm_dir() / "nvm.sh"


def check_nvm_installed() -> bool:
    """
    Check if nvm (Node Version Manager) is installed.
    """
    try:
        # Check if nvm is available in the shell
        # nvm is typically sourced in shell config files, so we need to check via shell
        if platform.system() == "Windows":
            # On Windows, nvm-windows uses a different approach
            result = subprocess.run(
                ["nvm", "version"],
                capture_output=True,
                text=True,
                timeout=10,
                shell=True,
            )
        else:
            # On Unix-like systems, nvm is a shell function, so we need to source it
            # First check if nvm.sh exists
            nvm_path = get_nvm_sh_path()
            if not nvm_path.exists():
                logger.debug(f"nvm.sh not found at {nvm_path}")
                return False

            # Try to source nvm and check version, capturing errors
            result = subprocess.run(
                ["bash", "-c", f"source {nvm_path} && nvm --version"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            if result.returncode != 0:
                # Log the error to help diagnose configuration issues
                if result.stderr:
                    logger.debug(f"nvm check failed: {result.stderr.strip()}")
                return False

        return result.returncode == 0
    except Exception as e:
        logger.debug(f"Exception checking nvm: {str(e)}")
        return False


def install_nvm() -> bool:
    """
    Install nvm (Node Version Manager) on Unix-like systems.
    """
    if platform.system() == "Windows":
        logger.error("nvm installation on Windows requires nvm-windows.")
        logger.error(
            "Please install nvm-windows manually from: https://github.com/coreybutler/nvm-windows"
        )
        return False

    logger.info("Installing nvm (Node Version Manager)...")

    try:
        # Download and install nvm
        nvm_install_script = "https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh"
        logger.info(f"Downloading nvm installer from {nvm_install_script}...")

        response = requests.get(nvm_install_script, timeout=60)
        response.raise_for_status()

        # Create a temporary script file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(response.text)
            install_script_path = f.name

        try:
            # Make the script executable and run it
            os.chmod(install_script_path, 0o755)
            result = subprocess.run(
                ["bash", install_script_path],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode == 0:
                logger.info("✓ nvm installed successfully")
                # Source nvm in current shell session
                nvm_dir = get_nvm_dir()
                if nvm_dir.exists():
                    return True
                else:
                    logger.warning(
                        f"nvm installation completed but nvm directory not found at {nvm_dir}"
                    )
                    return False
            else:
                logger.error(f"nvm installation failed: {result.stderr}")
                return False
        finally:
            # Clean up temporary script
            try:
                os.unlink(install_script_path)
            except Exception:
                pass

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download nvm installer: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Failed to install nvm: {str(e)}")
        return False


def install_node_with_nvm() -> bool:
    """
    Install the latest Node.js version using nvm.
    Returns True if installation succeeds, False otherwise.
    """
    if platform.system() == "Windows":
        logger.error("Node.js installation via nvm on Windows requires nvm-windows.")
        logger.error("Please install Node.js manually from: https://nodejs.org/")
        return False

    logger.info("Installing latest Node.js version using nvm...")

    try:
        # Source nvm and install latest Node.js
        nvm_path = get_nvm_sh_path()
        if not nvm_path.exists():
            logger.error(f"nvm.sh not found at {nvm_path}. nvm may not be properly installed.")
            return False

        nvm_source_cmd = f"source {nvm_path}"
        install_cmd = f"{nvm_source_cmd} && nvm install node"

        result = subprocess.run(
            ["bash", "-c", install_cmd],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes timeout for Node.js installation
        )

        if result.returncode == 0:
            logger.info("✓ Node.js installed successfully via nvm")

            # Set as default version
            use_cmd = f"{nvm_source_cmd} && nvm alias default node"
            subprocess.run(
                ["bash", "-c", use_cmd],
                capture_output=True,
                text=True,
                timeout=30,
            )

            # Add nvm to PATH for current session
            # This ensures node/npm are available in subsequent commands
            nvm_dir = get_nvm_dir()
            if nvm_dir.exists():
                # Update PATH for current process
                nvm_bin = nvm_dir / "versions" / "node"
                # Find the latest installed version
                if nvm_bin.exists():
                    versions = sorted(nvm_bin.iterdir(), reverse=True)
                    if versions:
                        latest_node_bin = versions[0] / "bin"
                        if latest_node_bin.exists():
                            current_path = os.environ.get("PATH", "")
                            os.environ["PATH"] = f"{latest_node_bin}:{current_path}"

            return True
        else:
            logger.error(f"Failed to install Node.js: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("Timeout installing Node.js (this can take several minutes)")
        return False
    except Exception as e:
        logger.error(f"Error installing Node.js: {str(e)}")
        return False


def check_node_npm() -> tuple[bool, str]:  # (is_available, error_message)
    """
    Check if Node.js and npm are available.
    If not available, attempts to install nvm and Node.js automatically.
    """

    try:
        # Check Node.js - try direct command first, then with nvm if needed
        result = subprocess.run(["node", "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            # If direct command fails, try with nvm sourced (in case nvm is installed but not in PATH)
            nvm_path = get_nvm_sh_path()
            if nvm_path.exists():
                result = subprocess.run(
                    ["bash", "-c", f"source {nvm_path} && node --version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode != 0 and result.stderr:
                    logger.debug(f"Failed to source nvm or run node: {result.stderr.strip()}")
        if result.returncode != 0:
            # Node.js is not installed, try to install it
            logger.info("Node.js is not installed. Attempting to install automatically...")

            # Check if nvm is installed
            if not check_nvm_installed():
                logger.info("nvm is not installed. Installing nvm first...")
                if not install_nvm():
                    return (
                        False,
                        "Failed to install nvm. Please install Node.js manually from https://nodejs.org/",
                    )

            # Install Node.js using nvm
            if not install_node_with_nvm():
                return (
                    False,
                    "Failed to install Node.js. Please install Node.js manually from https://nodejs.org/",
                )

            # Verify installation after automatic setup
            # Try with nvm sourced first
            nvm_path = get_nvm_sh_path()
            if nvm_path.exists():
                result = subprocess.run(
                    ["bash", "-c", f"source {nvm_path} && node --version"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode != 0 and result.stderr:
                    logger.debug(
                        f"Failed to verify node after installation: {result.stderr.strip()}"
                    )
            else:
                result = subprocess.run(
                    ["node", "--version"], capture_output=True, text=True, timeout=10
                )
            if result.returncode != 0:
                nvm_path = get_nvm_sh_path()
                return (
                    False,
                    f"Node.js installation completed but node command is not available. Please restart your terminal or source {nvm_path}",
                )

        node_version = result.stdout.strip()
        logger.debug(f"Found Node.js version: {node_version}")

        # Check npm - handle Windows PowerShell scripts
        if platform.system() == "Windows":
            # On Windows, npm might be a PowerShell script, so we need to use shell=True
            result = subprocess.run(
                ["npm", "--version"], capture_output=True, text=True, timeout=10, shell=True
            )
        else:
            # On Unix-like systems, if we just installed via nvm, we may need to source nvm
            # Try direct command first
            result = subprocess.run(
                ["npm", "--version"], capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                # Try with nvm sourced
                nvm_path = get_nvm_sh_path()
                if nvm_path.exists():
                    result = subprocess.run(
                        ["bash", "-c", f"source {nvm_path} && npm --version"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if result.returncode != 0 and result.stderr:
                        logger.debug(f"Failed to source nvm or run npm: {result.stderr.strip()}")

        if result.returncode != 0:
            return False, "npm is not installed or not in PATH"

        npm_version = result.stdout.strip()
        logger.debug(f"Found npm version: {npm_version}")

        return True, f"Node.js {node_version}, npm {npm_version}"

    except subprocess.TimeoutExpired:
        return False, "Timeout checking Node.js/npm installation"
    except FileNotFoundError:
        # Node.js is not installed, try to install it
        logger.info("Node.js is not found. Attempting to install automatically...")

        # Check if nvm is installed
        if not check_nvm_installed():
            logger.info("nvm is not installed. Installing nvm first...")
            if not install_nvm():
                return (
                    False,
                    "Failed to install nvm. Please install Node.js manually from https://nodejs.org/",
                )

        # Install Node.js using nvm
        if not install_node_with_nvm():
            return (
                False,
                "Failed to install Node.js. Please install Node.js manually from https://nodejs.org/",
            )

        # Retry checking Node.js after installation
        try:
            result = subprocess.run(
                ["node", "--version"], capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                node_version = result.stdout.strip()
                # Check npm
                nvm_path = get_nvm_sh_path()
                if nvm_path.exists():
                    result = subprocess.run(
                        ["bash", "-c", f"source {nvm_path} && npm --version"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )
                    if result.returncode == 0:
                        npm_version = result.stdout.strip()
                        return True, f"Node.js {node_version}, npm {npm_version}"
                    elif result.stderr:
                        logger.debug(f"Failed to source nvm or run npm: {result.stderr.strip()}")
        except Exception as e:
            logger.debug(f"Exception retrying node/npm check: {str(e)}")

        return False, "Node.js/npm not found. Please install Node.js from https://nodejs.org/"
    except Exception as e:
        return False, f"Error checking Node.js/npm: {str(e)}"
