"""PyPI connector unit tests: metadata fetching, document formatting, error handling, and dlt source."""

import json
from unittest.mock import MagicMock, patch
import urllib.error

import pytest

from cognee.tasks.ingestion.connectors.pypi import (
    PyPIError,
    PyPIPackageNotFoundError,
    PyPIService,
    build_pypi_service,
    pypi_source,
)
from cognee.tasks.ingestion.dlt_utils import DOCUMENT_SOURCE_ATTR, NODE_SET_COLUMN, PIPELINE_SCOPE_ATTR

MOCK_PACKAGE_DATA = {
    "info": {
        "name": "example-pkg",
        "version": "1.2.3",
        "summary": "An exemplary Python package for demonstration.",
        "author": "Test Author",
        "author_email": "test@example.com",
        "license": "MIT",
        "home_page": "https://example.com/example-pkg",
        "project_urls": {
            "Documentation": "https://example.com/docs",
            "Repository": "https://github.com/example/example-pkg",
        },
        "requires_python": ">=3.10",
        "requires_dist": [
            "pydantic>=2.0",
            "requests>=2.28; extra == 'network'",
        ],
        "classifiers": [
            "Development Status :: 4 - Beta",
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
        ],
        "description": "Long markdown description\n\n## Installation\n\n```bash\npip install example-pkg\n```",
    },
    "releases": {
        "1.0.0": [],
        "1.2.3": [{"upload_time_iso_8601": "2026-01-01T12:00:00Z"}],
    },
    "urls": [
        {"upload_time_iso_8601": "2026-01-01T12:00:00Z"},
    ],
}


def test_pypi_service_fetch_package_success():
    service = PyPIService(["example-pkg"])

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.read.return_value = json.dumps(MOCK_PACKAGE_DATA).encode("utf-8")
    mock_response.__enter__.return_value = mock_response

    with patch("urllib.request.urlopen", return_value=mock_response):
        data = service.fetch_package_metadata("example-pkg")

    assert data["info"]["name"] == "example-pkg"
    assert data["info"]["version"] == "1.2.3"


def test_pypi_service_fetch_package_not_found():
    service = PyPIService(["nonexistent-pkg-404"])

    http_error = urllib.error.HTTPError(
        url="https://pypi.org/pypi/nonexistent-pkg-404/json",
        code=404,
        msg="Not Found",
        hdrs={},
        fp=None,
    )

    with patch("urllib.request.urlopen", side_effect=http_error):
        with pytest.raises(PyPIPackageNotFoundError):
            service.fetch_package_metadata("nonexistent-pkg-404")


def test_pypi_format_package_document():
    service = PyPIService(["example-pkg"])
    doc = service.format_package_document(MOCK_PACKAGE_DATA)

    assert doc["id"] == "pypi:example-pkg"
    assert doc["title"] == "example-pkg 1.2.3"
    assert "https://example.com/example-pkg" in doc["url"]
    assert doc["_deleted"] is False
    assert "pypi" in doc[NODE_SET_COLUMN]
    assert "pypi:example-pkg" in doc[NODE_SET_COLUMN]

    content = doc["content"]
    assert "# example-pkg 1.2.3" in content
    assert "An exemplary Python package for demonstration." in content
    assert "MIT" in content
    assert ">=3.10" in content
    assert "pydantic>=2.0" in content
    assert "## Description" in content


def test_build_pypi_service_normalizes_list_or_str():
    s1 = build_pypi_service("requests")
    assert s1.packages == ["requests"]

    s2 = build_pypi_service(["cognee", "fastapi "])
    assert s2.packages == ["cognee", "fastapi"]


def test_pypi_source_metadata():
    pytest.importorskip("dlt")
    source = pypi_source(packages=["cognee", "fastapi"])

    assert getattr(source, DOCUMENT_SOURCE_ATTR) == "pypi"
    assert getattr(source, PIPELINE_SCOPE_ATTR) == "pypi_packages"
    assert source.name == "pypi_packages"
