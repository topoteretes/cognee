"""Media loaders explain themselves when cognee runs without an LLM key.

``add()`` stopped probing the LLM when none is configured, because keyless
cognee ingests text and documents on local models and the probe could only
restate the absent key. Media is the one ingestion path that still needs a key
— there is no local transcription or vision model — so each media loader has to
say so at the point it would have called the LLM, naming the file kind and what
still works without a key.
"""

import importlib

import pytest

from cognee.infrastructure.llm.exceptions import LLMAPIKeyNotSetError
from cognee.infrastructure.loaders.core.audio_loader import AudioLoader
from cognee.infrastructure.loaders.core.image_loader import ImageLoader
from cognee.infrastructure.loaders.core.video_loader import VideoLoader

require_llm_mod = importlib.import_module("cognee.infrastructure.loaders.utils.require_llm")
preflight_mod = importlib.import_module("cognee.modules.preflight")

MEDIA = [
    (ImageLoader, "photo.png", "Image"),
    (AudioLoader, "talk.mp3", "Audio"),
    (VideoLoader, "clip.mp4", "Video"),
]


@pytest.fixture
def no_llm(monkeypatch):
    monkeypatch.setattr(preflight_mod, "llm_available", lambda *_args, **_kwargs: False)


@pytest.fixture
def with_llm(monkeypatch):
    monkeypatch.setattr(preflight_mod, "llm_available", lambda *_args, **_kwargs: True)


@pytest.mark.parametrize(("loader_class", "filename", "media"), MEDIA)
@pytest.mark.asyncio
async def test_media_loader_names_the_key_it_needs(loader_class, filename, media, tmp_path, no_llm):
    media_file = tmp_path / filename
    media_file.write_bytes(b"pretend this is media")

    with pytest.raises(LLMAPIKeyNotSetError) as error:
        await loader_class().load(str(media_file))

    message = error.value.message
    assert media in message
    assert "LLM_API_KEY" in message
    # The half that still works keylessly is the actionable part: without it
    # the message reads as "cognee needs a key", which is not true any more.
    assert "need no key" in message


@pytest.mark.parametrize(("loader_class", "filename", "media"), MEDIA)
@pytest.mark.asyncio
async def test_missing_file_still_reports_the_missing_file(
    loader_class, filename, media, tmp_path, no_llm
):
    """The key check must not mask a plain typo in the path."""
    with pytest.raises(FileNotFoundError):
        await loader_class().load(str(tmp_path / f"absent-{filename}"))


@pytest.mark.parametrize(("loader_class", "filename", "media"), MEDIA)
@pytest.mark.asyncio
async def test_nothing_is_written_before_the_guard(loader_class, filename, media, tmp_path, no_llm):
    """The guard runs before the loader reads or stores anything."""
    media_file = tmp_path / filename
    media_file.write_bytes(b"pretend this is media")
    before = {path.name for path in tmp_path.iterdir()}

    with pytest.raises(LLMAPIKeyNotSetError):
        await loader_class().load(str(media_file))

    assert {path.name for path in tmp_path.iterdir()} == before


def test_guard_passes_when_an_llm_is_configured(with_llm):
    # Returns without raising: a configured key means the loader proceeds
    # exactly as it always did.
    assert require_llm_mod.require_llm_for_media("Image", "cognee describes images") is None
