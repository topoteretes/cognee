import pytest
from cognee.infrastructure.loaders.core.image_loader import ImageLoader


@pytest.fixture
def loader():
    return ImageLoader()


def test_image_loader_supported_extensions(loader):
    """Test that supported extensions are lowercase and have no leading dots."""
    extensions = loader.supported_extensions
    assert "jpeg" in extensions
    assert "jpe" in extensions
    assert "png" in extensions
    assert "jpg" in extensions

    for ext in extensions:
        assert not ext.startswith(".")
        assert ext == ext.lower()


@pytest.mark.parametrize(
    "extension, mime_type, expected",
    [
        ("png", "image/png", True),
        ("jpg", "image/jpeg", True),
        ("jpeg", "image/jpeg", True),
        ("jpe", "image/jpeg", True),
        ("txt", "text/plain", False),
        ("png", "text/plain", False),
    ],
)
def test_image_loader_can_handle(loader, extension, mime_type, expected):
    """Test that can_handle correctly identifies supported image formats."""
    assert loader.can_handle(extension, mime_type) == expected
