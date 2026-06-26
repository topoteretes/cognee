import os
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi import UploadFile

from cognee.infrastructure.files.storage.LocalFileStorage import LocalFileStorage
from cognee.modules.ingestion.save_data_to_file import save_data_to_file
from cognee.api.v1.ontologies.ontologies import OntologyService, OntologyMetadata


@pytest.mark.asyncio
async def test_local_file_storage_path_traversal():
    with tempfile.TemporaryDirectory() as temp_dir:
        storage = LocalFileStorage(temp_dir)

        # Valid operations should work
        safe_relative = "safe_folder/file.txt"
        await storage.store(safe_relative, "hello content", overwrite=True)
        assert await storage.file_exists(safe_relative) is True
        assert await storage.is_file(safe_relative) is True

        # List files relative check
        files = await storage.list_files("safe_folder")
        assert len(files) == 1
        assert files[0].endswith("safe_folder/file.txt")

        # Operations targeting parent directories should raise ValueError
        unsafe_paths = [
            "../unsafe.txt",
            "safe_folder/../../unsafe.txt",
            "/absolute/path/to/unsafe.txt",
        ]
        if os.name == "nt":
            unsafe_paths.append("C:\\Windows\\System32\\cmd.exe")

        for unsafe in unsafe_paths:
            with pytest.raises(ValueError, match="Path traversal detected"):
                await storage.store(unsafe, "data")

            with pytest.raises(ValueError, match="Path traversal detected"):
                async with storage.open(unsafe, "r"):
                    pass

            with pytest.raises(ValueError, match="Path traversal detected"):
                await storage.file_exists(unsafe)

            with pytest.raises(ValueError, match="Path traversal detected"):
                await storage.is_file(unsafe)

            with pytest.raises(ValueError, match="Path traversal detected"):
                await storage.get_size(unsafe)

            with pytest.raises(ValueError, match="Path traversal detected"):
                await storage.ensure_directory_exists(unsafe)

            with pytest.raises(ValueError, match="Path traversal detected"):
                await storage.remove(unsafe)

            with pytest.raises(ValueError, match="Path traversal detected"):
                await storage.list_files(unsafe)

            with pytest.raises(ValueError, match="Path traversal detected"):
                await storage.remove_all(unsafe)

            with pytest.raises(ValueError, match="Path traversal detected"):
                await storage.copy_file(unsafe, "dest.txt")

            with pytest.raises(ValueError, match="Path traversal detected"):
                await storage.copy_file("safe_folder/file.txt", unsafe)


@pytest.mark.asyncio
async def test_save_data_to_file_sanitizes_filename():
    # Setup temporary directories and configure mock storage config
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_config = {"data_root_directory": temp_dir}

        with patch(
            "cognee.modules.ingestion.save_data_to_file.get_storage_config",
            return_value=storage_config,
        ):
            # Verify save_data_to_file rejects names resolving to only dots or empty
            with pytest.raises(ValueError, match="Invalid filename"):
                with tempfile.SpooledTemporaryFile() as f:
                    f.write(b"some content")
                    f.seek(0)
                    await save_data_to_file(f, filename="../../../../")

            # Verify that path traversal in filenames gets stripped to the basename
            # e.g., "../../traversal_test.txt" -> "traversal_test.txt"
            with tempfile.SpooledTemporaryFile() as f:
                f.write(b"some content")
                f.seek(0)
                result_path_uri = await save_data_to_file(f, filename="../../traversal_test.txt")

            # The result is returned as a file URI (e.g. file:///...)
            # Convert URI back to local path and assert it is directly inside the temp_dir
            from urllib.parse import urlparse

            parsed_path = urlparse(result_path_uri).path
            if os.name == "nt" and parsed_path.startswith("/"):
                parsed_path = parsed_path.lstrip("/")

            resolved_file = Path(parsed_path).resolve()
            assert resolved_file.parent.resolve() == Path(temp_dir).resolve()
            assert resolved_file.name == "traversal_test.txt"


@pytest.mark.asyncio
async def test_ontology_service_path_traversal():
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock get_base_config for ontology root directory
        mock_config = MagicMock()
        mock_config.data_root_directory = Path(temp_dir)

        class MockUser:
            id = "test_user_uuid"

        user = MockUser()

        with patch("cognee.api.v1.ontologies.ontologies.get_base_config", return_value=mock_config):
            service = OntologyService()

            # Mock UploadFile with an async read method
            mock_file = MagicMock(spec=UploadFile)
            mock_file.filename = "test.owl"

            async def mock_read():
                return b"ontology data"

            mock_file.read = mock_read

            # 1. Test upload_ontology rejects unsafe keys
            unsafe_keys = ["../unsafe", "nested/../../unsafe", "user/keys"]
            for unsafe_key in unsafe_keys:
                with pytest.raises(ValueError, match="Invalid ontology key"):
                    await service.upload_ontology(unsafe_key, mock_file, user)

            # 2. Test get_ontology_contents rejects unsafe keys
            for unsafe_key in unsafe_keys:
                # We bypass metadata load check by mocking it since we want to verify path validation
                with patch.object(service, "_load_metadata", return_value={unsafe_key: {}}):
                    with pytest.raises(ValueError, match="Invalid ontology key"):
                        service.get_ontology_contents([unsafe_key], user)

            # 3. Test delete_ontology rejects unsafe keys
            for unsafe_key in unsafe_keys:
                with patch.object(service, "_load_metadata", return_value={unsafe_key: {}}):
                    with pytest.raises(ValueError, match="Invalid ontology key"):
                        service.delete_ontology(unsafe_key, user)
