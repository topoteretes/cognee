import pytest
import tempfile
import os
from unittest.mock import Mock, AsyncMock

from cognee.infrastructure.loaders.LoaderEngine import LoaderEngine
from cognee.infrastructure.loaders.LoaderInterface import LoaderInterface
from cognee.infrastructure.loaders.models.LoaderResult import LoaderResult, ContentType


class MockLoader(LoaderInterface):
    """Mock loader for testing."""

    def __init__(self, name="mock_loader", extensions=None, mime_types=None, fail_deps=False):
        self._name = name
        self._extensions = extensions or [".mock"]
        self._mime_types = mime_types or ["application/mock"]
        self._fail_deps = fail_deps

    @property
    def supported_extensions(self):
        return self._extensions

    @property
    def supported_mime_types(self):
        return self._mime_types

    @property
    def loader_name(self):
        return self._name

    def can_handle(self, file_path: str, mime_type: str = None) -> bool:
        ext = os.path.splitext(file_path)[1].lower()
        return ext in self._extensions or mime_type in self._mime_types

    async def load(self, file_path: str, **kwargs) -> LoaderResult:
        return LoaderResult(
            content=f"Mock content from {self._name}",
            metadata={"loader": self._name, "name": os.path.basename(file_path)},
            content_type=ContentType.TEXT,
        )

    def validate_dependencies(self) -> bool:
        return not self._fail_deps


class TestLoaderEngine:
    """Test the LoaderEngine class."""

    @pytest.fixture
    def engine(self):
        """Create a LoaderEngine instance for testing."""
        return LoaderEngine(
            loader_directories=[],
            default_loader_priority=["loader1", "loader2"],
            fallback_loader="fallback",
            enable_dependency_validation=True,
        )

    def test_engine_initialization(self, engine):
        """Test LoaderEngine initialization."""
        assert engine.loader_directories == []
        assert engine.default_loader_priority == ["loader1", "loader2"]
        assert engine.fallback_loader == "fallback"
        assert engine.enable_dependency_validation is True
        assert len(engine.get_available_loaders()) == 0

    def test_register_loader_success(self, engine):
        """Test successful loader registration."""
        loader = MockLoader("test_loader", [".test"])

        success = engine.register_loader(loader)

        assert success is True
        assert "test_loader" in engine.get_available_loaders()
        assert engine._loaders["test_loader"] == loader
        assert ".test" in engine._extension_map
        assert "application/mock" in engine._mime_type_map

    def test_register_loader_with_failed_dependencies(self, engine):
        """Test loader registration with failed dependency validation."""
        loader = MockLoader("test_loader", [".test"], fail_deps=True)

        success = engine.register_loader(loader)

        assert success is False
        assert "test_loader" not in engine.get_available_loaders()

    def test_register_loader_without_dependency_validation(self):
        """Test loader registration without dependency validation."""
        engine = LoaderEngine(
            loader_directories=[], default_loader_priority=[], enable_dependency_validation=False
        )
        loader = MockLoader("test_loader", [".test"], fail_deps=True)

        success = engine.register_loader(loader)

        assert success is True
        assert "test_loader" in engine.get_available_loaders()

    def test_get_loader_by_extension(self, engine):
        """Test getting loader by file extension."""
        loader1 = MockLoader("loader1", [".txt"])
        loader2 = MockLoader("loader2", [".pdf"])

        engine.register_loader(loader1)
        engine.register_loader(loader2)

        result = engine.get_loader("test.txt")
        assert result == loader1

        result = engine.get_loader("test.pdf")
        assert result == loader2

        result = engine.get_loader("test.unknown")
        assert result is None

    def test_get_loader_by_mime_type(self, engine):
        """Test getting loader by MIME type."""
        loader = MockLoader("loader", [".txt"], ["text/plain"])
        engine.register_loader(loader)

        result = engine.get_loader("test.unknown", mime_type="text/plain")
        assert result == loader

        result = engine.get_loader("test.unknown", mime_type="application/pdf")
        assert result is None

    def test_get_loader_with_preferences(self, engine):
        """Test getting loader with preferred loaders."""
        loader1 = MockLoader("loader1", [".txt"])
        loader2 = MockLoader("loader2", [".txt"])

        engine.register_loader(loader1)
        engine.register_loader(loader2)

        # Should get preferred loader
        result = engine.get_loader("test.txt", preferred_loaders=["loader2"])
        assert result == loader2

        # Should fallback to first available if preferred not found
        result = engine.get_loader("test.txt", preferred_loaders=["nonexistent"])
        assert result in [loader1, loader2]  # One of them should be returned

    def test_get_loader_with_priority(self, engine):
        """Test loader selection with priority order."""
        engine.default_loader_priority = ["priority_loader", "other_loader"]

        priority_loader = MockLoader("priority_loader", [".txt"])
        other_loader = MockLoader("other_loader", [".txt"])

        # Register in reverse order
        engine.register_loader(other_loader)
        engine.register_loader(priority_loader)

        # Should get priority loader even though other was registered first
        result = engine.get_loader("test.txt")
        assert result == priority_loader

    def test_get_loader_fallback(self, engine):
        """Test fallback loader selection."""
        fallback_loader = MockLoader("fallback", [".txt"])
        other_loader = MockLoader("other", [".pdf"])

        engine.register_loader(fallback_loader)
        engine.register_loader(other_loader)
        engine.fallback_loader = "fallback"

        # For .txt file, fallback should be considered
        result = engine.get_loader("test.txt")
        assert result == fallback_loader

        # For unknown extension, should still get fallback if it can handle
        result = engine.get_loader("test.unknown")
        assert result == fallback_loader

    @pytest.mark.asyncio
    async def test_load_file_success(self, engine):
        """Test successful file loading."""
        loader = MockLoader("test_loader", [".txt"])
        engine.register_loader(loader)

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test content")
            temp_path = f.name

        try:
            result = await engine.load_file(temp_path)
            assert result.content == "Mock content from test_loader"
            assert result.metadata["loader"] == "test_loader"
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_load_file_no_loader(self, engine):
        """Test file loading when no suitable loader is found."""
        with pytest.raises(ValueError, match="No loader found for file"):
            await engine.load_file("test.unknown")

    @pytest.mark.asyncio
    async def test_load_file_with_preferences(self, engine):
        """Test file loading with preferred loaders."""
        loader1 = MockLoader("loader1", [".txt"])
        loader2 = MockLoader("loader2", [".txt"])

        engine.register_loader(loader1)
        engine.register_loader(loader2)

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"test content")
            temp_path = f.name

        try:
            result = await engine.load_file(temp_path, preferred_loaders=["loader2"])
            assert result.metadata["loader"] == "loader2"
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_get_loader_info(self, engine):
        """Test getting loader information."""
        loader = MockLoader("test_loader", [".txt"], ["text/plain"])
        engine.register_loader(loader)

        info = engine.get_loader_info("test_loader")

        assert info["name"] == "test_loader"
        assert info["extensions"] == [".txt"]
        assert info["mime_types"] == ["text/plain"]
        assert info["available"] is True

        # Test non-existent loader
        info = engine.get_loader_info("nonexistent")
        assert info == {}

    def test_discover_loaders_empty_directory(self, engine):
        """Test loader discovery with empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            engine.loader_directories = [temp_dir]
            engine.discover_loaders()

            # Should not find any loaders in empty directory
            assert len(engine.get_available_loaders()) == 0

    def test_discover_loaders_nonexistent_directory(self, engine):
        """Test loader discovery with non-existent directory."""
        engine.loader_directories = ["/nonexistent/directory"]

        # Should not raise exception, just log warning
        engine.discover_loaders()
        assert len(engine.get_available_loaders()) == 0
