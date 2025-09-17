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


def delete_cache():
    """Delete the Cognee cache directory."""
    logger.info("Deleting cache...")
    cache_dir = get_cognee_cache_dir()
    if cache_dir.exists():
        for item in cache_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

    logger.info("✓ Cache deleted successfully!")


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


def _flatten_single_directory_extraction(cache_dir: Path) -> None:
    """
    If the extracted content consists of a single directory, flatten it by moving
    its contents to the parent level and removing the empty directory.
    Ignores common metadata directories like __MACOSX.

    Args:
        cache_dir: Directory containing extracted content
    """
    try:
        # Get all items in cache_dir (excluding metadata directories and any files we might have created)
        metadata_dirs = {"__MACOSX", ".DS_Store", "Thumbs.db"}
        relevant_items = [
            item
            for item in cache_dir.iterdir()
            if item.name not in metadata_dirs and not item.name.endswith((".txt", ".md"))
        ]

        # Check if there's exactly one relevant item and it's a directory
        if len(relevant_items) == 1 and relevant_items[0].is_dir():
            single_dir = relevant_items[0]

            # Move all contents of the single directory up one level
            temp_items = []
            for item in single_dir.iterdir():
                temp_name = f"_temp_{item.name}"
                temp_path = cache_dir / temp_name
                shutil.move(str(item), str(temp_path))
                temp_items.append((temp_path, cache_dir / item.name))

            # Remove the now-empty directory
            single_dir.rmdir()

            # Move items from temp names to final names
            for temp_path, final_path in temp_items:
                shutil.move(str(temp_path), str(final_path))

            logger.debug(f"Flattened single directory extraction in {cache_dir}")
    except Exception as e:
        logger.debug(f"Could not flatten directory structure: {e}")
        # Non-critical error, continue with nested structure


def download_and_extract_zip(
    url: str, cache_dir: Path, version_or_hash: str, force: bool = False
) -> None:
    """
    Download a zip file and extract it to cache directory with content freshness checking.

    Args:
        url: URL to download zip file from
        cache_dir: Directory to cache extracted content
        version_or_hash: Version string or content hash for cache validation
        force: If True, re-download even if already cached

    Returns:
        None
    """
    # Check if already cached and valid
    if not force and _is_cache_valid(cache_dir, version_or_hash):
        # Also check if remote content has changed
        is_fresh, new_identifier = _check_remote_content_freshness(url, cache_dir)
        if is_fresh:
            logger.debug(f"Content already cached and fresh for version {version_or_hash}")
        else:
            logger.info("Cached content is stale, updating...")
            # Don't clear cache yet, we'll overwrite it

    # Clear old cache if it exists
    if cache_dir.exists():
        _clear_cache(cache_dir)

    logger.info(f"Downloading content from {url}...")

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

        # Check if extraction created a single top-level directory and flatten if so
        _flatten_single_directory_extraction(cache_dir)

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


def get_tutorial_data_dir() -> Path:
    """Get the tutorial data cache directory."""
    return get_cache_subdir("tutorial_data")
