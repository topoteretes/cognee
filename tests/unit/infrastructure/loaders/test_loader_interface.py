import pytest
import tempfile
import os
from unittest.mock import AsyncMock

from cognee.infrastructure.loaders.LoaderInterface import LoaderInterface
from cognee.infrastructure.loaders.models.LoaderResult import LoaderResult, ContentType


class TestLoaderInterface:
    """Test the LoaderInterface abstract base class."""

    def test_loader_interface_is_abstract(self):
        """Test that LoaderInterface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LoaderInterface()

    def test_dependency_validation_with_no_dependencies(self):
        """Test dependency validation when no dependencies are required."""

        class MockLoader(LoaderInterface):
            @property
            def supported_extensions(self):
                return [".txt"]

            @property
            def supported_mime_types(self):
                return ["text/plain"]

            @property
            def loader_name(self):
                return "mock_loader"

            def can_handle(self, file_path: str, mime_type: str = None) -> bool:
                return True

            async def load(self, file_path: str, **kwargs) -> LoaderResult:
                return LoaderResult(content="test", metadata={}, content_type=ContentType.TEXT)

        loader = MockLoader()
        assert loader.validate_dependencies() is True
        assert loader.get_dependencies() == []

    def test_dependency_validation_with_missing_dependencies(self):
        """Test dependency validation with missing dependencies."""

        class MockLoaderWithDeps(LoaderInterface):
            @property
            def supported_extensions(self):
                return [".txt"]

            @property
            def supported_mime_types(self):
                return ["text/plain"]

            @property
            def loader_name(self):
                return "mock_loader_deps"

            def get_dependencies(self):
                return ["non_existent_package>=1.0.0"]

            def can_handle(self, file_path: str, mime_type: str = None) -> bool:
                return True

            async def load(self, file_path: str, **kwargs) -> LoaderResult:
                return LoaderResult(content="test", metadata={}, content_type=ContentType.TEXT)

        loader = MockLoaderWithDeps()
        assert loader.validate_dependencies() is False
        assert "non_existent_package>=1.0.0" in loader.get_dependencies()

    def test_dependency_validation_with_existing_dependencies(self):
        """Test dependency validation with existing dependencies."""

        class MockLoaderWithExistingDeps(LoaderInterface):
            @property
            def supported_extensions(self):
                return [".txt"]

            @property
            def supported_mime_types(self):
                return ["text/plain"]

            @property
            def loader_name(self):
                return "mock_loader_existing"

            def get_dependencies(self):
                return ["os"]  # Built-in module that always exists

            def can_handle(self, file_path: str, mime_type: str = None) -> bool:
                return True

            async def load(self, file_path: str, **kwargs) -> LoaderResult:
                return LoaderResult(content="test", metadata={}, content_type=ContentType.TEXT)

        loader = MockLoaderWithExistingDeps()
        assert loader.validate_dependencies() is True
