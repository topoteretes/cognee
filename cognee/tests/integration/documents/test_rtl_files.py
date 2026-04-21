import os
import pytest
import pathlib
import asyncio
from unittest.mock import patch, MagicMock
from cognee.base_config import get_base_config
from cognee.infrastructure.loaders.external.pypdf_loader import PyPdfLoader
from cognee.infrastructure.loaders.external.unstructured_loader import UnstructuredLoader

def clean_extracted_path(path: str) -> str:
    """Helper to handle Windows file:/// paths."""
    clean = path.replace("file:///", "").replace("file://", "")
    if ":" not in clean and clean.startswith("/"):
        clean = clean[1:]
    return clean

@pytest.mark.asyncio
async def test_rtl_pdf_extraction():
    test_file = pathlib.Path("cognee/tests/test_data/rtl/hebrew_alphabet.pdf")
    assert test_file.exists(), f"Test file {test_file} missing - should have been downloaded"

    config = get_base_config()
    original_flag = config.enable_rtl_support
    
    try:
        config.enable_rtl_support = True
        loader = PyPdfLoader()
        
        extracted_path = await loader.load(str(test_file))
        clean_path = clean_extracted_path(extracted_path)

        with open(clean_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Hebrew Alphabet chart should contain these
        assert "א" in content or "ב" in content
        
    finally:
        config.enable_rtl_support = original_flag

@pytest.mark.asyncio
async def test_rtl_docx_extraction():
    test_file = pathlib.Path("cognee/tests/test_data/rtl/hebrew_sample.docx")
    assert test_file.exists(), f"Test file {test_file} missing - should have been created"

    config = get_base_config()
    original_flag = config.enable_rtl_support
    
    try:
        config.enable_rtl_support = True
        loader = UnstructuredLoader()
        
        extracted_path = await loader.load(str(test_file))
        clean_path = clean_extracted_path(extracted_path)

        with open(clean_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        assert "היום בבוקר" in content
        assert "production" in content
        
    finally:
        config.enable_rtl_support = original_flag

@pytest.mark.asyncio
async def test_advanced_pdf_loader_rtl_coverage():
    from cognee.infrastructure.loaders.external.advanced_pdf_loader import AdvancedPdfLoader
    
    loader = AdvancedPdfLoader()
    config = get_base_config()
    original_flag = config.enable_rtl_support
    
    try:
        config.enable_rtl_support = True
        dummy_file = "dummy.pdf"
        with open(dummy_file, "wb") as f:
            f.write(b"%PDF-1.1\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << >> /Contents 4 0 R >>\nendobj\n4 0 obj\n<< /Length 0 >>\nstream\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f\n0000000009 00000 n\n0000000058 00000 n\n0000000115 00000 n\n0000000221 00000 n\ntrailer\n<< /Size 5 /Root 1 0 R >>\nstartxref\n272\n%%EOF")

        # Mock the entire unstructured.partition.pdf module
        mock_pdf_module = MagicMock()
        mock_pdf_module.partition_pdf.return_value = [MagicMock(category="Text", text="םולש")]
        
        with patch.dict("sys.modules", {"unstructured.partition.pdf": mock_pdf_module}):
            with patch("cognee.infrastructure.loaders.external.advanced_pdf_loader.get_file_metadata", return_value={"content_hash": "mock"}):
                with patch("cognee.infrastructure.loaders.external.advanced_pdf_loader.get_storage_config", return_value={"data_root_directory": ""}):
                    with patch("cognee.infrastructure.loaders.external.advanced_pdf_loader.get_file_storage") as mock_storage:
                        mock_storage_instance = MagicMock()
                        mock_storage.return_value = mock_storage_instance
                        async def mock_store(name, content): return "mock_path"
                        mock_storage_instance.store = MagicMock(side_effect=mock_store)
                        
                        await loader.load(dummy_file)
        
        if os.path.exists(dummy_file):
            os.remove(dummy_file)
    finally:
        config.enable_rtl_support = original_flag

@pytest.mark.asyncio
async def test_docling_loader_rtl_coverage():
    from cognee.infrastructure.loaders.external.docling_loader import DoclingLoader
    
    loader = DoclingLoader()
    config = get_base_config()
    original_flag = config.enable_rtl_support
    
    try:
        config.enable_rtl_support = True
        mock_conv_result = MagicMock()
        mock_conv_result.document.export_to_text.return_value = "םולש"
        
        dummy_file = "dummy_docling.pdf"
        with open(dummy_file, "wb") as f:
            f.write(b"test")

        with patch("cognee.infrastructure.loaders.external.docling_loader._get_docling_converter") as mock_get_conv:
            mock_get_conv.return_value.convert.return_value = mock_conv_result
            
            with patch("cognee.infrastructure.loaders.external.docling_loader.get_file_metadata", return_value={"content_hash": "mock"}):
                with patch("cognee.infrastructure.loaders.external.docling_loader.get_storage_config", return_value={"data_root_directory": ""}):
                    with patch("cognee.infrastructure.loaders.external.docling_loader.get_file_storage") as mock_storage:
                        mock_storage_instance = MagicMock()
                        mock_storage.return_value = mock_storage_instance
                        # Mock store as async
                        async def mock_store(name, content): return "mock_path"
                        mock_storage_instance.store = MagicMock(side_effect=mock_store)
                        
                        await loader.load(dummy_file)
        
        if os.path.exists(dummy_file):
            os.remove(dummy_file)
    finally:
        config.enable_rtl_support = original_flag
