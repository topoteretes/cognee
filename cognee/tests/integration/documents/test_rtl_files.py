import os
import pytest
import pathlib
from unittest.mock import patch, MagicMock
from cognee.base_config import get_base_config
from cognee.infrastructure.loaders.external.pypdf_loader import PyPdfLoader
from cognee.infrastructure.loaders.external.unstructured_loader import UnstructuredLoader

def clean_extracted_path(path: str) -> str:
    """Helper to handle Windows file:/// paths and plain filesystem paths."""
    if not path.startswith("file://"):
        return path
    clean = path.replace("file:///", "").replace("file://", "")
    # On Windows the residual path starts with a drive letter (e.g. "C:/...");
    # on POSIX re-prepend "/".
    if len(clean) >= 2 and clean[1] == ":":
        return clean
    return "/" + clean.lstrip("/")

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent

@pytest.mark.asyncio
async def test_rtl_pdf_extraction(monkeypatch):
    test_file = BASE_DIR / "test_data/rtl/hebrew_alphabet.pdf"
    assert test_file.exists(), f"Test file {test_file} missing - should have been downloaded"

    config = get_base_config()
    monkeypatch.setattr(config, "enable_rtl_support", True)
    
    loader = PyPdfLoader()
    
    # Patch maybe_normalize_rtl to verify it's called
    with patch("cognee.infrastructure.loaders.external.pypdf_loader.maybe_normalize_rtl",
               wraps=lambda x, **kw: x) as mock_normalize:
        extracted_path = await loader.load(str(test_file))
        mock_normalize.assert_called()

    clean_path = clean_extracted_path(extracted_path)

    with open(clean_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Hebrew Alphabet chart should contain these
    assert "א" in content or "ב" in content

@pytest.mark.asyncio
async def test_rtl_docx_extraction(monkeypatch):
    test_file = BASE_DIR / "test_data/rtl/hebrew_sample.docx"
    assert test_file.exists(), f"Test file {test_file} missing - should have been created"

    config = get_base_config()
    monkeypatch.setattr(config, "enable_rtl_support", True)
    
    loader = UnstructuredLoader()
    extracted_path = await loader.load(str(test_file))
    clean_path = clean_extracted_path(extracted_path)

    with open(clean_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    assert "היום בבוקר" in content
    assert "production" in content

@pytest.mark.asyncio
async def test_advanced_pdf_loader_rtl_coverage(monkeypatch, tmp_path):
    from cognee.infrastructure.loaders.external.advanced_pdf_loader import AdvancedPdfLoader
    
    loader = AdvancedPdfLoader()
    config = get_base_config()
    monkeypatch.setattr(config, "enable_rtl_support", True)
    
    dummy_file = tmp_path / "dummy.pdf"
    
    pdf_content = (
        b"%PDF-1.1\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << >> /Contents 4 0 R >>\nendobj\n"
        b"4 0 obj\n<< /Length 0 >>\nstream\nendstream\nendobj\n"
        b"xref\n0 5\n0000000000 65535 f\n0000000009 00000 n\n"
        b"0000000058 00000 n\n0000000115 00000 n\n0000000221 00000 n\n"
        b"trailer\n<< /Size 5 /Root 1 0 R >>\n"
        b"startxref\n272\n%%EOF"
    )
    
    with open(dummy_file, "wb") as f:
        f.write(pdf_content)

    # Mock the entire unstructured.partition.pdf module
    mock_pdf_module = MagicMock()
    mock_pdf_module.partition_pdf.return_value = [MagicMock(category="Text", text="םולש")]
    
    async def mock_store(name, content):
        return "mock_path"

    with (
        patch.dict("sys.modules", {"unstructured.partition.pdf": mock_pdf_module}),
        patch("cognee.infrastructure.loaders.external.advanced_pdf_loader.get_file_metadata",
               return_value={"content_hash": "mock"}),
        patch("cognee.infrastructure.loaders.external.advanced_pdf_loader.get_storage_config",
               return_value={"data_root_directory": ""}),
        patch("cognee.infrastructure.loaders.external.advanced_pdf_loader.get_file_storage") as mock_storage,
        patch("cognee.infrastructure.loaders.external.advanced_pdf_loader.maybe_normalize_rtl",
               wraps=lambda x, **kw: x) as mock_normalize
    ):
        mock_storage_instance = MagicMock()
        mock_storage.return_value = mock_storage_instance
        mock_storage_instance.store = MagicMock(side_effect=mock_store)
        
        await loader.load(str(dummy_file))
        mock_normalize.assert_called()

@pytest.mark.asyncio
async def test_docling_loader_rtl_coverage(monkeypatch, tmp_path):
    from cognee.infrastructure.loaders.external.docling_loader import DoclingLoader
    
    loader = DoclingLoader()
    config = get_base_config()
    monkeypatch.setattr(config, "enable_rtl_support", True)
    
    mock_conv_result = MagicMock()
    mock_conv_result.document.export_to_text.return_value = "םולש"
    
    dummy_file = tmp_path / "dummy_docling.pdf"
    with open(dummy_file, "wb") as f:
        f.write(b"test")
        
    async def mock_store(name, content):
        return "mock_path"

    with (
        patch("cognee.infrastructure.loaders.external.docling_loader._get_docling_converter") as mock_get_conv,
        patch("cognee.infrastructure.loaders.external.docling_loader.get_file_metadata",
               return_value={"content_hash": "mock"}),
        patch("cognee.infrastructure.loaders.external.docling_loader.get_storage_config",
               return_value={"data_root_directory": ""}),
        patch("cognee.infrastructure.loaders.external.docling_loader.get_file_storage") as mock_storage,
        patch("cognee.infrastructure.loaders.external.docling_loader.maybe_normalize_rtl",
               wraps=lambda x, **kw: x) as mock_normalize
    ):
        mock_get_conv.return_value = mock_converter = MagicMock()
        mock_converter.convert.return_value = mock_conv_result
        
        mock_storage_instance = MagicMock()
        mock_storage.return_value = mock_storage_instance
        mock_storage_instance.store = MagicMock(side_effect=mock_store)
        
        await loader.load(str(dummy_file))
        mock_normalize.assert_called()

@pytest.mark.asyncio
async def test_rtl_processor_direct_logic():
    # These tests exercise the `cognee[rtl]` optional dependencies.
    pytest.importorskip("bidi.algorithm")
    pytest.importorskip("arabic_reshaper")
    
    from cognee.modules.data.processing.rtl_processor import process_rtl_text
    
    # Visual string (reversed Hebrew, but LTR English)
    # Visual: "םולש Hello" -> Logical: "Hello שלום"
    visual_input = "םולש Hello"
    
    # When disabled, should be identical
    assert process_rtl_text(visual_input, enable_rtl=False) == visual_input
    
    # When enabled and is_visual=True (explicit; default is None for auto-detection)
    result = process_rtl_text(visual_input, enable_rtl=True, is_visual=True)
    assert result == "Hello שלום"
    
    # When enabled but is_visual=False, it should do nothing
    logical_input = "היום בבוקר 2/10 מהDBים של postgress נפלו בproduction"
    assert process_rtl_text(logical_input, enable_rtl=True, is_visual=False) == logical_input
