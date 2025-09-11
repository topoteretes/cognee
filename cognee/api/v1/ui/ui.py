import os
import signal
import subprocess
import threading
import time
import webbrowser
import zipfile
import requests
from pathlib import Path
from typing import Callable, Optional, Tuple
import tempfile
import shutil

from cognee.shared.logging_utils import get_logger
from cognee.version import get_cognee_version

logger = get_logger()


def normalize_version_for_comparison(version: str) -> str:
    """
    Normalize version string for comparison.
    Handles development versions and edge cases.
    """
    # Remove common development suffixes for comparison
    normalized = (
        version.replace("-local", "").replace("-dev", "").replace("-alpha", "").replace("-beta", "")
    )
    return normalized.strip()


def get_frontend_cache_dir() -> Path:
    """
    Get the directory where downloaded frontend assets are cached.
    Uses user's home directory to persist across package updates.
    Each cached frontend is version-specific and will be re-downloaded
    when the cognee package version changes.
    """
    cache_dir = Path.home() / ".cognee" / "ui-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_frontend_download_info() -> Tuple[str, str]:
    """
    Get the download URL and version for the actual cognee-frontend source.
    Downloads the real frontend from GitHub releases, matching the installed version.
    """
    version = get_cognee_version()

    # Clean up version string (remove -local suffix for development)
    clean_version = version.replace("-local", "")

    # Download from specific release tag to ensure version compatibility
    download_url = f"https://github.com/topoteretes/cognee/archive/refs/tags/v{clean_version}.zip"

    return download_url, version


def download_frontend_assets(force: bool = False) -> bool:
    """
    Download and cache frontend assets.

    Args:
        force: If True, re-download even if already cached

    Returns:
        bool: True if successful, False otherwise
    """
    cache_dir = get_frontend_cache_dir()
    frontend_dir = cache_dir / "frontend"
    version_file = cache_dir / "version.txt"

    # Check if already downloaded and up to date
    if not force and frontend_dir.exists() and version_file.exists():
        try:
            cached_version = version_file.read_text().strip()
            current_version = get_cognee_version()

            # Compare normalized versions to handle development versions
            cached_normalized = normalize_version_for_comparison(cached_version)
            current_normalized = normalize_version_for_comparison(current_version)

            if cached_normalized == current_normalized:
                logger.debug(f"Frontend assets already cached for version {current_version}")
                return True
            else:
                logger.info(
                    f"Version mismatch detected: cached={cached_version}, current={current_version}"
                )
                logger.info("Updating frontend cache to match current cognee version...")
                # Clear the old cached version
                if frontend_dir.exists():
                    shutil.rmtree(frontend_dir)
                if version_file.exists():
                    version_file.unlink()
        except Exception as e:
            logger.debug(f"Error checking cached version: {e}")
            # Clear potentially corrupted cache
            if frontend_dir.exists():
                shutil.rmtree(frontend_dir)
            if version_file.exists():
                version_file.unlink()

    download_url, version = get_frontend_download_info()

    logger.info(f"Downloading cognee frontend assets for version {version}...")
    logger.info("This will be cached and reused until the cognee version changes.")

    try:
        # Create a temporary directory for download
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            archive_path = temp_path / "cognee-main.zip"

            # Download the actual cognee repository from releases
            logger.info(
                f"Downloading cognee v{version.replace('-local', '')} from GitHub releases..."
            )
            logger.info(f"URL: {download_url}")
            response = requests.get(download_url, stream=True, timeout=60)
            response.raise_for_status()

            with open(archive_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Extract the archive and find the cognee-frontend directory
            if frontend_dir.exists():
                shutil.rmtree(frontend_dir)

            with zipfile.ZipFile(archive_path, "r") as zip_file:
                # Extract to temp directory first
                extract_dir = temp_path / "extracted"
                zip_file.extractall(extract_dir)

                # Find the cognee-frontend directory in the extracted content
                # The archive structure will be: cognee-{version}/cognee-frontend/
                cognee_frontend_source = None
                for root, dirs, files in os.walk(extract_dir):
                    if "cognee-frontend" in dirs:
                        cognee_frontend_source = Path(root) / "cognee-frontend"
                        break

                if not cognee_frontend_source or not cognee_frontend_source.exists():
                    logger.error(
                        "Could not find cognee-frontend directory in downloaded release archive"
                    )
                    logger.error("This might indicate a version mismatch or missing release.")
                    return False

                # Copy the cognee-frontend to our cache
                shutil.copytree(cognee_frontend_source, frontend_dir)
                logger.debug(f"Frontend extracted to: {frontend_dir}")

            # Write version info for future cache validation
            version_file.write_text(version)
            logger.debug(f"Cached frontend for cognee version: {version}")

            logger.info(
                f"✓ Cognee frontend v{version.replace('-local', '')} downloaded and cached successfully!"
            )
            return True

    except requests.exceptions.RequestException as e:
        if "404" in str(e):
            logger.error(f"Release v{version.replace('-local', '')} not found on GitHub.")
            logger.error(
                "This version might not have been released yet, or you're using a development version."
            )
            logger.error("Try using a stable release version of cognee.")
        else:
            logger.error(f"Failed to download from GitHub: {str(e)}")
        logger.error("You can still use cognee without the UI functionality.")
        return False
    except Exception as e:
        logger.error(f"Failed to download frontend assets: {str(e)}")
        logger.error("You can still use cognee without the UI functionality.")
        return False


def find_frontend_path() -> Optional[Path]:
    """
    Find the cognee-frontend directory.
    Checks both development location and cached download location.
    """
    current_file = Path(__file__)

    # First, try development paths (for contributors/developers)
    dev_search_paths = [
        current_file.parents[4] / "cognee-frontend",  # from cognee/api/v1/ui/ui.py to project root
        current_file.parents[3] / "cognee-frontend",  # fallback path
        current_file.parents[2] / "cognee-frontend",  # another fallback
    ]

    for path in dev_search_paths:
        if path.exists() and (path / "package.json").exists():
            logger.debug(f"Found development frontend at: {path}")
            return path

    # Then try cached download location (for pip-installed users)
    cache_dir = get_frontend_cache_dir()
    cached_frontend = cache_dir / "frontend"

    if cached_frontend.exists() and (cached_frontend / "package.json").exists():
        logger.debug(f"Found cached frontend at: {cached_frontend}")
        return cached_frontend

    return None


def check_node_npm() -> tuple[bool, str]:
    """
    Check if Node.js and npm are available.
    Returns (is_available, error_message)
    """
    try:
        # Check Node.js
        result = subprocess.run(["node", "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return False, "Node.js is not installed or not in PATH"

        node_version = result.stdout.strip()
        logger.debug(f"Found Node.js version: {node_version}")

        # Check npm
        result = subprocess.run(["npm", "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return False, "npm is not installed or not in PATH"

        npm_version = result.stdout.strip()
        logger.debug(f"Found npm version: {npm_version}")

        return True, f"Node.js {node_version}, npm {npm_version}"

    except subprocess.TimeoutExpired:
        return False, "Timeout checking Node.js/npm installation"
    except FileNotFoundError:
        return False, "Node.js/npm not found. Please install Node.js from https://nodejs.org/"
    except Exception as e:
        return False, f"Error checking Node.js/npm: {str(e)}"


def install_frontend_dependencies(frontend_path: Path) -> bool:
    """
    Install frontend dependencies if node_modules doesn't exist.
    This is needed for both development and downloaded frontends since both use npm run dev.
    """
    node_modules = frontend_path / "node_modules"
    if node_modules.exists():
        logger.debug("Frontend dependencies already installed")
        return True

    logger.info("Installing frontend dependencies (this may take a few minutes)...")

    try:
        result = subprocess.run(
            ["npm", "install"],
            cwd=frontend_path,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes timeout
        )

        if result.returncode == 0:
            logger.info("Frontend dependencies installed successfully")
            return True
        else:
            logger.error(f"Failed to install dependencies: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        logger.error("Timeout installing frontend dependencies")
        return False
    except Exception as e:
        logger.error(f"Error installing frontend dependencies: {str(e)}")
        return False


def is_development_frontend(frontend_path: Path) -> bool:
    """
    Check if this is a development frontend (has Next.js) vs downloaded assets.
    """
    package_json_path = frontend_path / "package.json"
    if not package_json_path.exists():
        return False

    try:
        import json

        with open(package_json_path) as f:
            package_data = json.load(f)

        # Development frontend has Next.js as dependency
        dependencies = package_data.get("dependencies", {})
        dev_dependencies = package_data.get("devDependencies", {})

        return "next" in dependencies or "next" in dev_dependencies
    except Exception:
        return False


def prompt_user_for_download() -> bool:
    """
    Ask user if they want to download the frontend assets.
    Returns True if user consents, False otherwise.
    """
    try:
        print("\n" + "=" * 60)
        print("🎨 Cognee UI Setup Required")
        print("=" * 60)
        print("The cognee frontend is not available on your system.")
        print("This is required to use the web interface.")
        print("\nWhat will happen:")
        print("• Download the actual cognee-frontend from GitHub")
        print("• Cache it in your home directory (~/.cognee/ui-cache/)")
        print("• Install dependencies with npm (requires Node.js)")
        print("• This is a one-time setup per cognee version")
        print("\nThe frontend will then be available offline for future use.")

        response = input("\nWould you like to download the frontend now? (y/N): ").strip().lower()
        return response in ["y", "yes"]
    except (KeyboardInterrupt, EOFError):
        print("\nOperation cancelled by user.")
        return False


def start_ui(
    pid_callback: Callable[[int], None],
    host: str = "localhost",
    port: int = 3000,
    open_browser: bool = True,
    auto_download: bool = False,
    start_backend: bool = False,
    backend_host: str = "localhost",
    backend_port: int = 8000,
) -> Optional[subprocess.Popen]:
    """
    Start the cognee frontend UI server, optionally with the backend API server.

    This function will:
    1. Optionally start the cognee backend API server
    2. Find the cognee-frontend directory (development) or download it (pip install)
    3. Check if Node.js and npm are available (for development mode)
    4. Install dependencies if needed (development mode)
    5. Start the frontend server
    6. Optionally open the browser

    Args:
        pid_callback: Callback to notify with PID of each spawned process
        host: Host to bind the frontend server to (default: localhost)
        port: Port to run the frontend server on (default: 3000)
        open_browser: Whether to open the browser automatically (default: True)
        auto_download: If True, download frontend without prompting (default: False)
        start_backend: If True, also start the cognee API backend server (default: False)
        backend_host: Host to bind the backend server to (default: localhost)
        backend_port: Port to run the backend server on (default: 8000)

    Returns:
        subprocess.Popen object representing the running frontend server, or None if failed
        Note: If backend is started, it runs in a separate process that will be cleaned up
        when the frontend process is terminated.

    Example:
        >>> import cognee
        >>> # Start just the frontend
        >>> server = cognee.start_ui()
        >>>
        >>> # Start both frontend and backend
        >>> server = cognee.start_ui(start_backend=True)
        >>> # UI will be available at http://localhost:3000
        >>> # API will be available at http://localhost:8000
        >>> # To stop both servers later:
        >>> server.terminate()
    """
    logger.info("Starting cognee UI...")
    backend_process = None

    # Start backend server if requested
    if start_backend:
        logger.info("Starting cognee backend API server...")
        try:
            import sys

            backend_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "cognee.api.client:app",
                    "--host",
                    backend_host,
                    "--port",
                    str(backend_port),
                ],
                # Inherit stdout/stderr from parent process to show logs
                stdout=None,
                stderr=None,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )

            pid_callback(backend_process.pid)

            # Give the backend a moment to start
            time.sleep(2)

            if backend_process.poll() is not None:
                logger.error("Backend server failed to start - process exited early")
                return None

            logger.info(f"✓ Backend API started at http://{backend_host}:{backend_port}")

        except Exception as e:
            logger.error(f"Failed to start backend server: {str(e)}")
            return None

    # Find frontend directory
    frontend_path = find_frontend_path()

    if not frontend_path:
        logger.info("Frontend not found locally. This is normal for pip-installed cognee.")

        # Offer to download the frontend
        if auto_download or prompt_user_for_download():
            if download_frontend_assets():
                frontend_path = find_frontend_path()
                if not frontend_path:
                    logger.error(
                        "Download succeeded but frontend still not found. This is unexpected."
                    )
                    return None
            else:
                logger.error("Failed to download frontend assets.")
                return None
        else:
            logger.info("Frontend download declined. UI functionality not available.")
            logger.info("You can still use all other cognee features without the web interface.")
            return None

    # Check Node.js and npm
    node_available, node_message = check_node_npm()
    if not node_available:
        logger.error(f"Cannot start UI: {node_message}")
        logger.error("Please install Node.js from https://nodejs.org/ to use the UI functionality")
        return None

    logger.debug(f"Environment check passed: {node_message}")

    # Install dependencies if needed
    if not install_frontend_dependencies(frontend_path):
        logger.error("Failed to install frontend dependencies")
        return None

    # Prepare environment variables
    env = os.environ.copy()
    env["HOST"] = host
    env["PORT"] = str(port)

    # Start the development server
    logger.info(f"Starting frontend server at http://{host}:{port}")
    logger.info("This may take a moment to compile and start...")

    try:
        # Create frontend in its own process group for clean termination
        process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=frontend_path,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None,
        )

        pid_callback(process.pid)

        # Give it a moment to start up
        time.sleep(3)

        # Check if process is still running
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            logger.error("Frontend server failed to start:")
            logger.error(f"stdout: {stdout}")
            logger.error(f"stderr: {stderr}")
            return None

        # Open browser if requested
        if open_browser:

            def open_browser_delayed():
                time.sleep(5)  # Give Next.js time to fully start
                try:
                    webbrowser.open(f"http://{host}:{port}")  # TODO: use dashboard url?
                except Exception as e:
                    logger.warning(f"Could not open browser automatically: {e}")

            browser_thread = threading.Thread(target=open_browser_delayed, daemon=True)
            browser_thread.start()

        logger.info("✓ Cognee UI is starting up...")
        logger.info(f"✓ Open your browser to: http://{host}:{port}")
        logger.info("✓ The UI will be available once Next.js finishes compiling")

        # Store backend process reference in the frontend process for cleanup
        if backend_process:
            process._cognee_backend_process = backend_process

        return process

    except Exception as e:
        logger.error(f"Failed to start frontend server: {str(e)}")
        # Clean up backend process if it was started
        if backend_process:
            logger.info("Cleaning up backend process due to frontend failure...")
            try:
                backend_process.terminate()
                backend_process.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError, ProcessLookupError):
                try:
                    backend_process.kill()
                    backend_process.wait()
                except (OSError, ProcessLookupError):
                    pass
        return None


def stop_ui(process: subprocess.Popen) -> bool:
    """
    Stop a running UI server process and backend process (if started), along with all their children.

    Args:
        process: The subprocess.Popen object returned by start_ui()

    Returns:
        bool: True if stopped successfully, False otherwise
    """
    if not process:
        return False

    success = True

    try:
        # First, stop the backend process if it exists
        backend_process = getattr(process, "_cognee_backend_process", None)
        if backend_process:
            logger.info("Stopping backend server...")
            try:
                backend_process.terminate()
                try:
                    backend_process.wait(timeout=5)
                    logger.info("Backend server stopped gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning("Backend didn't terminate gracefully, forcing kill")
                    backend_process.kill()
                    backend_process.wait()
                    logger.info("Backend server stopped")
            except Exception as e:
                logger.error(f"Error stopping backend server: {str(e)}")
                success = False

        # Now stop the frontend process
        logger.info("Stopping frontend server...")
        # Try to terminate the process group (includes child processes like Next.js)
        if hasattr(os, "killpg"):
            try:
                # Kill the entire process group
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                logger.debug("Sent SIGTERM to process group")
            except (OSError, ProcessLookupError):
                # Fall back to terminating just the main process
                process.terminate()
                logger.debug("Terminated main process only")
        else:
            process.terminate()
            logger.debug("Terminated main process (Windows)")

        try:
            process.wait(timeout=10)
            logger.info("Frontend server stopped gracefully")
        except subprocess.TimeoutExpired:
            logger.warning("Frontend didn't terminate gracefully, forcing kill")

            # Force kill the process group
            if hasattr(os, "killpg"):
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    logger.debug("Sent SIGKILL to process group")
                except (OSError, ProcessLookupError):
                    process.kill()
                    logger.debug("Force killed main process only")
            else:
                process.kill()
                logger.debug("Force killed main process (Windows)")

            process.wait()

        if success:
            logger.info("UI servers stopped successfully")

        return success

    except Exception as e:
        logger.error(f"Error stopping UI servers: {str(e)}")
        return False


# Convenience function similar to DuckDB's approach
def ui() -> Optional[subprocess.Popen]:
    """
    Convenient alias for start_ui() with default parameters.
    Similar to how DuckDB provides simple ui() function.
    """
    return start_ui()


if __name__ == "__main__":
    # Test the UI startup
    server = start_ui()
    if server:
        try:
            input("Press Enter to stop the server...")
        finally:
            stop_ui(server)
