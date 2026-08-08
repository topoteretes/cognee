import pytest
from cognee.infrastructure.loaders.core.image_loader import ImageLoader


@pytest.fixture
def loader():
    return ImageLoader()


def test_supported_extensions_no_leading_dots(loader):
    extensions = loader.supported_extensions
    for ext in extensions:
        assert not ext.startswith("."), f"Extension '{ext}' has leading dot"


def test_supported_extensions_lowercase(loader):
    extensions = loader.supported_extensions
    for ext in extensions:
        assert ext == ext.lower(), f"Extension '{ext}' is not lowercase"


def test_jpeg_formats_present(loader):
    extensions = loader.supported_extensions
    assert "jpeg" in extensions
    assert "jpe" in extensions
    assert "jpg" in extensions


def test_common_formats_present(loader):
    extensions = loader.supported_extensions
    for fmt in ["png", "gif", "webp", "bmp", "tiff", "ico"]:
        assert fmt in extensions, f"Missing extension: {fmt}"


@pytest.mark.parametrize(
    "extension, mime_type, expected",
    [
        ("png", "image/png", True),
        ("jpg", "image/jpeg", True),
        ("jpeg", "image/jpeg", True),
        ("jpe", "image/jpeg", True),
        ("gif", "image/gif", True),
        ("webp", "image/webp", True),
        ("txt", "text/plain", False),
        ("png", "text/plain", False),
        ("pdf", "application/pdf", False),
        ("jpeg", "text/plain", False),
    ],
)
def test_can_handle(loader, extension, mime_type, expected):
    assert loader.can_handle(extension, mime_type) == expected


@pytest.mark.parametrize(
    "extension, mime_type",
    [
        ("JPEG", "image/jpeg"),
        ("JPG", "image/jpeg"),
        ("Png", "image/png"),
    ],
)
def test_can_handle_case_sensitivity(loader, extension, mime_type):
    result = loader.can_handle(extension, mime_type)
    assert result is False, f"can_handle should be case-sensitive, got True for {extension}"
