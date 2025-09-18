"""
Storage-aware cache management utilities for Cognee.

This module provides cache functionality that works with both local and cloud storage
backends (like S3) through the StorageManager abstraction.
"""

import hashlib
import zipfile
import asyncio
from typing import Optional, Tuple
import aiohttp
import logging
from io import BytesIO

from cognee.base_config import get_base_config
from cognee.infrastructure.files.storage.get_file_storage import get_file_storage
from cognee.infrastructure.files.storage.StorageManager import StorageManager
from cognee.shared.utils import create_secure_ssl_context

logger = logging.getLogger(__name__)


class StorageAwareCache:
    """
    A cache manager that works with different storage backends (local, S3, etc.)
    """

    def __init__(self, cache_subdir: str = "cache"):
        """
        Initialize the cache manager.

        Args:
            cache_subdir: Subdirectory name within the system root for caching
        """
        self.base_config = get_base_config()
        # Since we're using cache_root_directory, don't add extra cache prefix
        self.cache_base_path = ""
        self.storage_manager: StorageManager = get_file_storage(
            self.base_config.cache_root_directory
        )

        # Print absolute path
        storage_path = self.storage_manager.storage.storage_path
        if storage_path.startswith("s3://"):
            absolute_path = storage_path  # S3 paths are already absolute
        else:
            import os

            absolute_path = os.path.abspath(storage_path)
        logger.info(f"Storage manager absolute path: {absolute_path}")

    async def get_cache_dir(self) -> str:
        """Get the base cache directory path."""
        cache_path = self.cache_base_path or "."  # Use "." for root when cache_base_path is empty
        await self.storage_manager.ensure_directory_exists(cache_path)
        return cache_path

    async def get_cache_subdir(self, name: str) -> str:
        """Get a specific cache subdirectory."""
        if self.cache_base_path:
            cache_path = f"{self.cache_base_path}/{name}"
        else:
            cache_path = name
        await self.storage_manager.ensure_directory_exists(cache_path)

        # Return the absolute path based on storage system
        if self.storage_manager.storage.storage_path.startswith("s3://"):
            return cache_path
        elif hasattr(self.storage_manager.storage, "storage_path"):
            return f"{self.storage_manager.storage.storage_path}/{cache_path}"
        else:
            # Fallback for other storage types
            return cache_path

    async def delete_cache(self):
        """Delete the entire cache directory."""
        logger.info("Deleting cache...")
        try:
            await self.storage_manager.remove_all(self.cache_base_path)
            logger.info("✓ Cache deleted successfully!")
        except Exception as e:
            logger.error(f"Error deleting cache: {e}")
            raise

    async def _is_cache_valid(self, cache_dir: str, version_or_hash: str) -> bool:
        """Check if cached content is valid for the given version/hash."""
        version_file = f"{cache_dir}/version.txt"

        if not await self.storage_manager.file_exists(version_file):
            return False

        try:
            async with self.storage_manager.open(version_file, "r") as f:
                cached_version = (await asyncio.to_thread(f.read)).strip()
                return cached_version == version_or_hash
        except Exception as e:
            logger.debug(f"Error checking cache validity: {e}")
            return False

    async def _clear_cache(self, cache_dir: str) -> None:
        """Clear a cache directory."""
        try:
            await self.storage_manager.remove_all(cache_dir)
        except Exception as e:
            logger.debug(f"Error clearing cache directory {cache_dir}: {e}")

    async def _check_remote_content_freshness(
        self, url: str, cache_dir: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if remote content is fresher than cached version using HTTP headers.

        Returns:
            Tuple of (is_fresh: bool, new_identifier: Optional[str])
        """
        try:
            # Make a HEAD request to check headers without downloading
            ssl_context = create_secure_ssl_context()
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.head(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
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
            identifier_file = f"{cache_dir}/content_id.txt"
            if await self.storage_manager.file_exists(identifier_file):
                async with self.storage_manager.open(identifier_file, "r") as f:
                    cached_identifier = (await asyncio.to_thread(f.read)).strip()
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

    async def download_and_extract_zip(
        self, url: str, cache_subdir_name: str, version_or_hash: str, force: bool = False
    ) -> str:
        """
        Download a zip file and extract it to cache directory with content freshness checking.

        Args:
            url: URL to download zip file from
            cache_subdir_name: Name of the cache subdirectory
            version_or_hash: Version string or content hash for cache validation
            force: If True, re-download even if already cached

        Returns:
            Path to the cached directory
        """
        cache_dir = await self.get_cache_subdir(cache_subdir_name)

        # Check if already cached and valid
        if not force and await self._is_cache_valid(cache_dir, version_or_hash):
            # Also check if remote content has changed
            is_fresh, new_identifier = await self._check_remote_content_freshness(url, cache_dir)
            if is_fresh:
                logger.debug(f"Content already cached and fresh for version {version_or_hash}")
                return cache_dir
            else:
                logger.info("Cached content is stale, updating...")

        # Clear old cache if it exists
        await self._clear_cache(cache_dir)

        logger.info(f"Downloading content from {url}...")

        # Download the zip file
        zip_content = BytesIO()
        etag = ""
        last_modified = ""
        ssl_context = create_secure_ssl_context()
        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as response:
                response.raise_for_status()

                # Extract headers before consuming response
                etag = response.headers.get("ETag", "").strip('"')
                last_modified = response.headers.get("Last-Modified", "")

                # Read the response content
                async for chunk in response.content.iter_chunked(8192):
                    zip_content.write(chunk)
        zip_content.seek(0)

        # Extract the archive
        await self.storage_manager.ensure_directory_exists(cache_dir)

        # Extract files and store them using StorageManager
        with zipfile.ZipFile(zip_content, "r") as zip_file:
            for file_info in zip_file.infolist():
                if file_info.is_dir():
                    # Create directory
                    dir_path = f"{cache_dir}/{file_info.filename}"
                    await self.storage_manager.ensure_directory_exists(dir_path)
                else:
                    # Extract and store file
                    file_data = zip_file.read(file_info.filename)
                    file_path = f"{cache_dir}/{file_info.filename}"
                    await self.storage_manager.store(file_path, BytesIO(file_data), overwrite=True)

        # Write version info for future cache validation
        version_file = f"{cache_dir}/version.txt"
        await self.storage_manager.store(version_file, version_or_hash, overwrite=True)

        # Store content identifier from response headers for freshness checking
        content_identifier = etag if etag else last_modified

        if content_identifier:
            identifier_file = f"{cache_dir}/content_id.txt"
            await self.storage_manager.store(identifier_file, content_identifier, overwrite=True)
            logger.debug(f"Stored content identifier: {content_identifier[:20]}...")

        logger.info("✓ Content downloaded and cached successfully!")
        return cache_dir

    async def file_exists(self, file_path: str) -> bool:
        """Check if a file exists in cache storage."""
        return await self.storage_manager.file_exists(file_path)

    async def read_file(self, file_path: str, encoding: str = "utf-8"):
        """Read a file from cache storage."""
        return self.storage_manager.open(file_path, encoding=encoding)

    async def list_files(self, directory_path: str):
        """List files in a cache directory."""
        try:
            file_list = await self.storage_manager.list_files(directory_path)

            # For S3 storage, convert relative paths to full S3 URLs
            if self.storage_manager.storage.storage_path.startswith("s3://"):
                full_paths = []
                for file_path in file_list:
                    full_s3_path = f"{self.storage_manager.storage.storage_path}/{file_path}"
                    full_paths.append(full_s3_path)
                return full_paths
            else:
                # For local storage, return absolute paths
                storage_path = self.storage_manager.storage.storage_path
                if not storage_path.startswith("/"):
                    import os

                    storage_path = os.path.abspath(storage_path)

                full_paths = []
                for file_path in file_list:
                    if file_path.startswith("/"):
                        full_paths.append(file_path)  # Already absolute
                    else:
                        full_paths.append(f"{storage_path}/{file_path}")
                return full_paths

        except Exception as e:
            logger.debug(f"Error listing files in {directory_path}: {e}")
            return []


# Convenience functions that maintain API compatibility
_cache_manager = None


def get_cache_manager() -> StorageAwareCache:
    """Get a singleton cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = StorageAwareCache()
    return _cache_manager


def generate_content_hash(url: str, additional_data: str = "") -> str:
    """Generate a content hash from URL and optional additional data."""
    content = f"{url}:{additional_data}"
    return hashlib.md5(content.encode()).hexdigest()[:12]  # Short hash for readability


# Async wrapper functions for backward compatibility
async def delete_cache():
    """Delete the Cognee cache directory."""
    cache_manager = get_cache_manager()
    await cache_manager.delete_cache()


async def get_cognee_cache_dir() -> str:
    """Get the base Cognee cache directory."""
    cache_manager = get_cache_manager()
    return await cache_manager.get_cache_dir()


async def get_cache_subdir(name: str) -> str:
    """Get a specific cache subdirectory."""
    cache_manager = get_cache_manager()
    return await cache_manager.get_cache_subdir(name)


async def download_and_extract_zip(
    url: str, cache_dir_name: str, version_or_hash: str, force: bool = False
) -> str:
    """Download a zip file and extract it to cache directory."""
    cache_manager = get_cache_manager()
    return await cache_manager.download_and_extract_zip(url, cache_dir_name, version_or_hash, force)


async def get_tutorial_data_dir() -> str:
    """Get the tutorial data cache directory."""
    return await get_cache_subdir("tutorial_data")


# Cache file operations
async def cache_file_exists(file_path: str) -> bool:
    """Check if a file exists in cache storage."""
    cache_manager = get_cache_manager()
    return await cache_manager.file_exists(file_path)


async def read_cache_file(file_path: str, encoding: str = "utf-8"):
    """Read a file from cache storage."""
    cache_manager = get_cache_manager()
    return await cache_manager.read_file(file_path, encoding)


async def list_cache_files(directory_path: str):
    """List files in a cache directory."""
    cache_manager = get_cache_manager()
    return await cache_manager.list_files(directory_path)
