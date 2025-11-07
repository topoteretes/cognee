import os
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest
from PIL import Image

from cognee.base_config import get_base_config
from cognee.infrastructure.loaders.core.image_loader import ImageLoader
from cognee.infrastructure.mineru.http_client import MineruHTTPClientError


def _prepare_environment(tmp_path: Path):
    data_root = tmp_path / "data"
    system_root = tmp_path / "system"
    os.environ["DATA_ROOT_DIRECTORY"] = str(data_root)
    os.environ["SYSTEM_ROOT_DIRECTORY"] = str(system_root)
    get_base_config.cache_clear()


def _create_image(path: Path):
    image = Image.new("RGB", (10, 10), color=(123, 222, 100))
    image.save(path, format="PNG")


def _read_text_file(path: str) -> str:
    if path.startswith("file://"):
        parsed = urlparse(path)
        return Path(parsed.path).read_text(encoding="utf-8")
    return Path(path).read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_image_loader_prefers_mineru(monkeypatch, tmp_path):
    _prepare_environment(tmp_path)

    from cognee.infrastructure import mineru as mineru_module

    mineru_module.get_mineru_http_client.cache_clear()

    sample_path = tmp_path / "sample.png"
    _create_image(sample_path)

    class DummyMineruClient:
        def __init__(self):
            self.calls = 0
            self.last_source = None

        async def extract_text(self, image_bytes: bytes, *, prompt=None, source_name=None):
            self.calls += 1
            self.last_source = source_name
            return "mineru output"

    dummy_client = DummyMineruClient()

    async def fail_fallback(*_args, **_kwargs):
        raise AssertionError("Fallback should not be invoked when MinerU succeeds.")

    monkeypatch.setattr(
        "cognee.infrastructure.loaders.core.image_loader.get_mineru_http_client",
        lambda: dummy_client,
    )
    monkeypatch.setattr(
        "cognee.infrastructure.llm.LLMGateway.transcribe_image",
        staticmethod(fail_fallback),
    )

    loader = ImageLoader()
    output_path = await loader.load(str(sample_path))

    assert _read_text_file(output_path) == "mineru output"

    assert dummy_client.calls == 1
    assert dummy_client.last_source == sample_path.name


@pytest.mark.asyncio
async def test_image_loader_falls_back_when_mineru_fails(monkeypatch, tmp_path):
    _prepare_environment(tmp_path)

    from cognee.infrastructure import mineru as mineru_module

    mineru_module.get_mineru_http_client.cache_clear()

    sample_path = tmp_path / "sample.png"
    _create_image(sample_path)

    class FailingClient:
        async def extract_text(self, *_args, **_kwargs):
            raise MineruHTTPClientError("network error")

    async def fallback_response(*_args, **_kwargs):
        message = SimpleNamespace(content="fallback output")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    monkeypatch.setattr(
        "cognee.infrastructure.loaders.core.image_loader.get_mineru_http_client",
        lambda: FailingClient(),
    )
    monkeypatch.setattr(
        "cognee.infrastructure.llm.LLMGateway.transcribe_image",
        staticmethod(fallback_response),
    )

    loader = ImageLoader()
    output_path = await loader.load(str(sample_path))

    assert _read_text_file(output_path) == "fallback output"
