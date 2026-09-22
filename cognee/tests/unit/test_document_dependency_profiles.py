"""Keep the document extras and MCP's default install independent of Pandoc."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
from packaging.requirements import Requirement

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[3]


def read_toml(path):
    return tomllib.loads((ROOT / path).read_text())


@pytest.fixture
def unstructured_loader():
    # Load an isolated module so a stub partitioner cannot leak into other tests.
    auto = ModuleType("unstructured.partition.auto")
    auto.partition = lambda **kwargs: pytest.fail("Routing must not convert documents")
    spec = importlib.util.spec_from_file_location(
        "routing_test_unstructured",
        ROOT / "cognee/infrastructure/loaders/external/unstructured_loader.py",
    )
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {"unstructured.partition.auto": auto}):
        spec.loader.exec_module(module)
    return module.UnstructuredLoader()


def test_docs_extra_has_no_pandoc_converters():
    docs = read_toml("pyproject.toml")["project"]["optional-dependencies"]["docs"]
    requirement = next(Requirement(value) for value in docs if value.startswith("unstructured["))
    assert requirement.extras.isdisjoint({"epub", "odt", "org", "rst", "rtf"})
    assert {"docx", "pptx", "xlsx", "pdf"} <= requirement.extras
    names = {package["name"] for package in read_toml("uv.lock")["package"]}
    assert names.isdisjoint({"pypandoc", "pypandoc-binary"})


def test_mcp_defaults_to_slim_docling():
    project = read_toml("cognee-mcp/pyproject.toml")["project"]
    requirements = [Requirement(value) for value in project["dependencies"]]
    cognee = next(value for value in requirements if value.name == "cognee")
    assert "docling" in cognee.extras
    assert "docs" not in cognee.extras
    names = {package["name"] for package in read_toml("cognee-mcp/uv.lock")["package"]}
    assert {"docling-slim", "odfdo"} <= names
    assert names.isdisjoint({"unstructured", "pypandoc", "pypandoc-binary", "torch"})
    assert not any(name.startswith("nvidia-") for name in names)


@pytest.mark.parametrize(
    "extension,mime",
    [
        ("odt", "application/vnd.oasis.opendocument.text"),
        ("rtf", "application/rtf"),
        ("epub", "application/epub+zip"),
    ],
)
def test_pandoc_formats_route_past_unstructured(extension, mime, monkeypatch, unstructured_loader):
    from cognee.infrastructure.loaders.LoaderEngine import LoaderEngine

    unstructured = unstructured_loader
    assert not unstructured.can_handle(extension, mime)
    engine = LoaderEngine()
    engine.register_loader(unstructured)
    docling = SimpleNamespace(
        loader_name="docling_loader",
        supported_extensions=[extension],
        supported_mime_types=[mime],
        can_handle=lambda **kwargs: True,
    )
    engine.register_loader(docling)
    monkeypatch.setattr(
        "cognee.infrastructure.loaders.LoaderEngine.guess_file_type",
        lambda path: SimpleNamespace(extension=extension, mime=mime),
    )
    assert engine.get_loader(f"example.{extension}", None) is docling


def test_unstructured_keeps_office_formats(unstructured_loader):
    loader = unstructured_loader
    for extension, kind in [
        ("docx", "wordprocessingml.document"),
        ("pptx", "presentationml.presentation"),
        ("xlsx", "spreadsheetml.sheet"),
    ]:
        assert loader.can_handle(extension, f"application/vnd.openxmlformats-officedocument.{kind}")
