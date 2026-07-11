from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock

import pytest

from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.infrastructure.loaders.core.image_loader import ImageLoader

TEST_DATA = Path(__file__).parent / "test_data"

MOCK_CAPTION = "A description of the image from the VLM."

OCR_GROUND_TRUTH = {
    "revenue_table.png": "QuarterlyRevenue\nQ1\n120K\nQ2\n150K\nQ3\n210K",
    # "lig ht bullb" is a minor recognition quirk of the pinned model — kept as the honest gold.
    "example.png": (
        "How many programmers does\n"
        "it take to change a lig ht bullb?\n"
        "None. That's a hardware\n"
        "problem."
    ),
}


def _mock_transcription(text):
    """Vision-LLM response exposing `.choices[0].message.content` (what ImageLoader reads)."""
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


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


@pytest.mark.parametrize(
    "env_value, expected",
    [
        (None, False),
        ("false", False),
        ("true", True),
        ("TRUE", True),
    ],
)
def test_ocr_enabled_reads_env(loader, monkeypatch, env_value, expected):
    """_ocr_enabled reads IMAGE_OCR_ENABLED (case-insensitive 'true'), defaulting to off."""
    if env_value is None:
        monkeypatch.delenv("IMAGE_OCR_ENABLED", raising=False)
    else:
        monkeypatch.setenv("IMAGE_OCR_ENABLED", env_value)
    assert loader._ocr_enabled() is expected


@pytest.mark.asyncio
@pytest.mark.parametrize("fixture_name", ["revenue_table.png", "example.png"])
async def test_load_appends_real_ocr_text(loader, monkeypatch, fixture_name):
    """With OCR enabled, real text read from the image is appended to the vision caption.

    Only the LLM call is mocked (no API key); rapidocr runs for real over a fixture image.
    """
    monkeypatch.setenv("IMAGE_OCR_ENABLED", "true")

    with patch.object(
        LLMGateway,
        "transcribe_image",
        new=AsyncMock(return_value=_mock_transcription(MOCK_CAPTION)),
    ):
        content = await loader.load(str(TEST_DATA / fixture_name), persist=False)

    expected = f"{MOCK_CAPTION}\n\n[OCR extracted text]\n{OCR_GROUND_TRUTH[fixture_name]}"
    assert content == expected, f"{content = } != {expected = }"


@pytest.mark.asyncio
async def test_load_omits_ocr_when_disabled(loader, monkeypatch):
    """With OCR disabled (default), the loader output is the vision caption only."""
    monkeypatch.delenv("IMAGE_OCR_ENABLED", raising=False)

    with patch.object(
        LLMGateway,
        "transcribe_image",
        new=AsyncMock(return_value=_mock_transcription(MOCK_CAPTION)),
    ):
        content = await loader.load(str(TEST_DATA / "revenue_table.png"), persist=False)

    assert content == MOCK_CAPTION, f"{content = } != {MOCK_CAPTION = }"


@pytest.mark.asyncio
async def test_load_skips_ocr_section_when_no_text(loader, monkeypatch):
    """OCR enabled but the image yields no text: the caption is returned with no OCR section."""
    monkeypatch.setenv("IMAGE_OCR_ENABLED", "true")

    with (
        patch.object(
            LLMGateway,
            "transcribe_image",
            new=AsyncMock(return_value=_mock_transcription(MOCK_CAPTION)),
        ),
        patch.object(ImageLoader, "_extract_ocr_text", new=AsyncMock(return_value="")),
    ):
        content = await loader.load(str(TEST_DATA / "revenue_table.png"), persist=False)

    assert content == MOCK_CAPTION, f"{content = } != {MOCK_CAPTION = }"
