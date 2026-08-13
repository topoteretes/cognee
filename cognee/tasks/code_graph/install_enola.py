"""Download and install a pinned enola release on first use.

The enola binary (https://github.com/enola-labs/enola, Apache-2.0 by Enola
Labs) is a compiled Go CLI that cannot be installed from PyPI. When it is not
already available, cognee downloads the pinned release for the current
platform straight from the author's GitHub releases, verifies it against the
SHA-256 checksums pinned below, and installs it under ``~/.cognee/bin``.

Controls:

- ``ENOLA_AUTO_INSTALL`` (default ``true``): set to ``false`` to disable the
  automatic download and get the manual-install error instead.
- ``ALLOW_HTTP_REQUESTS`` (default ``true``): cognee's global outbound-HTTP
  switch is honored; when false the download is refused.
- ``ENOLA_PATH``: always wins over any auto-installed binary.
"""

import hashlib
import os
import platform
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Optional

from fastapi import status

from cognee.exceptions import CogneeSystemError
from cognee.shared.logging_utils import get_logger

logger = get_logger("enola")

ENOLA_PINNED_VERSION = "0.3.13"

_RELEASE_URL_TEMPLATE = "https://github.com/enola-labs/enola/releases/download/v{version}/{asset}"

# SHA-256 of each release archive, pinned from the .sha256 files published
# alongside the v0.3.13 release assets. Bumping ENOLA_PINNED_VERSION requires
# re-pinning these.
ENOLA_RELEASE_CHECKSUMS = {
    "darwin-arm64": "f395dfaa213a816539f4178d1519bd13117764eb9b0cd0029b3d319040c0aba7",
    "darwin-amd64": "50292c297831a2e77fd6fb9617f307219141496fa4cea09dd64511e7919291dd",
    "linux-amd64": "870cb18f589620171b195812aa44a9883b7234206fbab95d42dc20069942ed9e",
    "linux-arm64": "bb86c297a3f6a0071a6a4b4c180e08bf836e42c4e98fab258d9d763ad626e2b0",
    "windows-amd64": "8d4a0a03b1f17af7f5174722e542fa06da2b8101d4f6fd09537e6d176b774a17",
}

_FALSEY = {"false", "0", "no", "off"}

_DOWNLOAD_TIMEOUT_SECONDS = 120


class EnolaInstallError(CogneeSystemError):
    def __init__(
        self,
        message: str = "Automatic enola installation failed.",
        name: str = "EnolaInstallError",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
    ):
        super().__init__(message, name, status_code)


def auto_install_enabled() -> bool:
    """Whether the automatic download may run (ENOLA_AUTO_INSTALL, default true)."""
    return os.getenv("ENOLA_AUTO_INSTALL", "true").strip().lower() not in _FALSEY


def _http_requests_allowed() -> bool:
    return os.getenv("ALLOW_HTTP_REQUESTS", "true").strip().lower() not in _FALSEY


def _platform_key() -> str:
    """The release asset key for this machine, e.g. 'darwin-arm64'."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    arch = {"x86_64": "amd64", "amd64": "amd64", "arm64": "arm64", "aarch64": "arm64"}.get(machine)
    key = f"{system}-{arch}" if arch else None
    if key not in ENOLA_RELEASE_CHECKSUMS:
        raise EnolaInstallError(
            message=(
                f"No pinned enola v{ENOLA_PINNED_VERSION} build for platform "
                f"'{system}/{machine}'. Install enola manually and set ENOLA_PATH."
            )
        )
    return key


def installed_binary_path() -> Path:
    """Where the auto-installed binary lives (may not exist yet)."""
    suffix = ".exe" if platform.system().lower() == "windows" else ""
    binary_name = f"enola-{ENOLA_PINNED_VERSION}-{_platform_key()}{suffix}"
    return Path.home() / ".cognee" / "bin" / binary_name


def _download(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:
        with open(destination, "wb") as archive_file:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                archive_file.write(chunk)


def _extract_single_binary(archive_path: Path, destination: Path) -> None:
    """Extract the archive's single enola binary member without trusting member paths.

    Releases up to 0.1.x shipped the binary alone; 0.3.x adds LICENSE and
    NOTICE next to it. Exactly one top-level `enola*` member must exist, and
    no member name may contain a path separator or start with a dot.
    """
    with tarfile.open(archive_path, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isreg()]
        if any("/" in member.name or member.name.startswith(".") for member in members):
            names = [member.name for member in archive.getmembers()]
            raise EnolaInstallError(
                message=f"Unexpected enola archive layout {names}; refusing to extract."
            )
        binaries = [member for member in members if member.name.startswith("enola")]
        if len(binaries) != 1:
            names = [member.name for member in archive.getmembers()]
            raise EnolaInstallError(
                message=f"Unexpected enola archive layout {names}; refusing to extract."
            )
        extracted = archive.extractfile(binaries[0])
        if extracted is None:
            raise EnolaInstallError(
                message=f"Could not read '{binaries[0].name}' from the enola archive."
            )
        with open(destination, "wb") as binary_file:
            binary_file.write(extracted.read())


def install_enola(install_dir: Optional[Path] = None) -> str:
    """Install the pinned enola release and return the binary path.

    Idempotent: returns the existing binary without any network access when it
    is already installed. The downloaded archive must match the checksum
    pinned in ENOLA_RELEASE_CHECKSUMS or nothing is installed.
    """
    platform_key = _platform_key()
    target = (
        Path(install_dir) / installed_binary_path().name if install_dir else installed_binary_path()
    )
    if target.is_file():
        return str(target)

    if not _http_requests_allowed():
        raise EnolaInstallError(
            message=(
                "Cannot auto-install enola: outbound HTTP requests are disabled "
                "(ALLOW_HTTP_REQUESTS=false). Install enola manually and set ENOLA_PATH."
            )
        )

    asset = f"enola-{ENOLA_PINNED_VERSION}-{platform_key}.tar.gz"
    url = _RELEASE_URL_TEMPLATE.format(version=ENOLA_PINNED_VERSION, asset=asset)
    expected_checksum = ENOLA_RELEASE_CHECKSUMS[platform_key]

    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading enola v%s (%s) from %s", ENOLA_PINNED_VERSION, platform_key, url)

    with tempfile.TemporaryDirectory(dir=target.parent) as temp_dir:
        archive_path = Path(temp_dir) / asset
        try:
            _download(url, archive_path)
        except OSError as error:
            raise EnolaInstallError(
                message=f"Failed to download enola from {url}: {error}"
            ) from error

        actual_checksum = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        if actual_checksum != expected_checksum:
            raise EnolaInstallError(
                message=(
                    f"Checksum mismatch for {asset}: expected {expected_checksum}, "
                    f"got {actual_checksum}. Refusing to install."
                )
            )

        staged_binary = Path(temp_dir) / target.name
        _extract_single_binary(archive_path, staged_binary)
        staged_binary.chmod(0o755)
        # Atomic within the same directory tree, so a concurrent installer
        # can never observe a half-written binary.
        os.replace(staged_binary, target)

    logger.info("Installed enola v%s at %s", ENOLA_PINNED_VERSION, target)
    return str(target)
