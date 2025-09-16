"""
Shared cache management utilities for Cognee.

This module provides common functionality for managing cached resources
in the user's ~/.cognee directory, including version tracking, downloads,
and file extraction.
"""

import hashlib
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Tuple
import requests
import logging

logger = logging.getLogger(__name__)


def get_cognee_cache_dir() -> Path:
    """Get the base Cognee cache directory (~/.cognee)."""
    cache_dir = Path.home() / ".cognee"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def get_cache_subdir(name: str) -> Path:
    """Get a specific cache subdirectory within ~/.cognee/."""
    cache_dir = get_cognee_cache_dir() / name
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def generate_content_hash(url: str, additional_data: str = "") -> str:
    """Generate a content hash from URL and optional additional data."""
    content = f"{url}:{additional_data}"
    return hashlib.md5(content.encode()).hexdigest()[:12]  # Short hash for readability


def _is_cache_valid(cache_dir: Path, version_or_hash: str) -> bool:
    """Check if cached content is valid for the given version/hash."""
    version_file = cache_dir / "version.txt"
    if not cache_dir.exists() or not version_file.exists():
        return False

    try:
        cached_version = version_file.read_text().strip()
        return cached_version == version_or_hash
    except Exception as e:
        logger.debug(f"Error checking cache validity: {e}")
        return False


def _clear_cache(cache_dir: Path) -> None:
    """Clear a cache directory."""
    if cache_dir.exists():
        shutil.rmtree(cache_dir)


def _check_remote_content_freshness(url: str, cache_dir: Path) -> Tuple[bool, Optional[str]]:
    """
    Check if remote content is fresher than cached version using HTTP headers.

    Args:
        url: URL to check
        cache_dir: Cache directory containing version.txt and etag.txt

    Returns:
        Tuple of (is_fresh: bool, new_identifier: Optional[str])
        is_fresh = True means cache is still valid
        new_identifier = ETag or Last-Modified value if different
    """
    try:
        # Make a HEAD request to check headers without downloading
        response = requests.head(url, timeout=30)
        response.raise_for_status()

        # Try ETag first (most reliable)
        etag = response.headers.get("ETag", "").strip('"')
        last_modified = response.headers.get("Last-Modified", "")

        # Use ETag if available, otherwise Last-Modified
        remote_identifier = etag if etag else last_modified

        if not remote_identifier:
            logger.debug("No freshness headers available, cannot check for updates")
            return True, None  # Assume fresh if no headers

        # Check cached identifier
        identifier_file = cache_dir / "content_id.txt"
        if identifier_file.exists():
            cached_identifier = identifier_file.read_text().strip()
            if cached_identifier == remote_identifier:
                logger.debug(f"Content is fresh (identifier: {remote_identifier[:20]}...)")
                return True, None
            else:
                logger.info(
                    f"Content has changed (old: {cached_identifier[:20]}..., new: {remote_identifier[:20]}...)"
                )
                return False, remote_identifier
        else:
            # No cached identifier, treat as stale
            return False, remote_identifier

    except Exception as e:
        logger.debug(f"Could not check remote freshness: {e}")
        return True, None  # Assume fresh if we can't check


def download_and_extract_zip(
    url: str, cache_dir: Path, version_or_hash: str, force: bool = False
) -> Tuple[bool, Optional[Path]]:
    """
    Download a zip file and extract it to cache directory with content freshness checking.

    Args:
        url: URL to download zip file from
        cache_dir: Directory to cache extracted content
        version_or_hash: Version string or content hash for cache validation
        force: If True, re-download even if already cached

    Returns:
        Tuple of (success: bool, extracted_path: Optional[Path])
    """
    # Check if already cached and valid
    if not force and _is_cache_valid(cache_dir, version_or_hash):
        # Also check if remote content has changed
        is_fresh, new_identifier = _check_remote_content_freshness(url, cache_dir)
        if is_fresh:
            logger.debug(f"Content already cached and fresh for version {version_or_hash}")
            return True, cache_dir
        else:
            logger.info("Cached content is stale, updating...")
            # Don't clear cache yet, we'll overwrite it

    # Clear old cache if it exists
    if cache_dir.exists():
        _clear_cache(cache_dir)

    logger.info(f"Downloading content from {url}...")

    try:
        # Create a temporary directory for download
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            archive_path = temp_path / "download.zip"

            # Download the zip file
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()

            with open(archive_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            # Extract the archive
            cache_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(archive_path, "r") as zip_file:
                zip_file.extractall(cache_dir)

            # Write version info for future cache validation
            version_file = cache_dir / "version.txt"
            version_file.write_text(version_or_hash)

            # Store content identifier from response headers for freshness checking
            etag = response.headers.get("ETag", "").strip('"')
            last_modified = response.headers.get("Last-Modified", "")
            content_identifier = etag if etag else last_modified

            if content_identifier:
                identifier_file = cache_dir / "content_id.txt"
                identifier_file.write_text(content_identifier)
                logger.debug(f"Stored content identifier: {content_identifier[:20]}...")

            logger.info("✓ Content downloaded and cached successfully!")
            return True, cache_dir

    except requests.exceptions.RequestException as e:
        if "404" in str(e):
            logger.error(f"Content not found at {url}")
        else:
            logger.error(f"Failed to download from {url}: {str(e)}")
        return False, None
    except Exception as e:
        logger.error(f"Failed to download and extract content: {str(e)}")
        return False, None


def get_tutorial_data_dir() -> Path:
    """Get the tutorial data cache directory."""
    return get_cache_subdir("tutorial_data")


def store_tutorial_files(source_dir: Path, data_files_dir: Path) -> bool:
    """
    Store tutorial data files in the cache directory.

    Args:
        source_dir: Directory containing tutorial files to store
        data_files_dir: Target directory to store files

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if data_files_dir.exists():
            shutil.rmtree(data_files_dir)

        # Look for a 'data' directory in the source
        source_data_dir = source_dir / "data"
        if source_data_dir.exists():
            shutil.copytree(source_data_dir, data_files_dir)
            logger.info(f"Tutorial data files stored in {data_files_dir}")
            return True
        else:
            logger.debug("No data directory found in tutorial zip")
            data_files_dir.mkdir(parents=True, exist_ok=True)
            return True

    except Exception as e:
        logger.error(f"Failed to store tutorial files: {e}")
        return False
