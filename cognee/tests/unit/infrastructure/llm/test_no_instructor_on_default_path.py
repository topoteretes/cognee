"""The default LLM path (``STRUCTURED_OUTPUT_FRAMEWORK=litellm_native``) never imports
the legacy ``instructor`` package.

Runs in a subprocess: the shared pytest interpreter already has the package loaded by
the legacy adapter tests, so a ``sys.modules`` assertion is only meaningful in a fresh
interpreter. The child imports the public entry points and the media loaders, then
drives both ``LLMGateway`` transcription entry points against a mocked litellm boundary.
"""

import os
import subprocess
import sys
import textwrap

import pytest

_CHILD = textwrap.dedent(
    """
    import asyncio, os, sys, tempfile
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, patch

    def legacy_modules():
        return sorted(m for m in sys.modules if m == "instructor" or m.startswith("instructor."))

    import cognee  # noqa: F401
    from cognee.infrastructure.llm.LLMGateway import LLMGateway
    import cognee.api.v1.remember.remember  # noqa: F401
    import cognee.api.v1.recall.recall  # noqa: F401
    import cognee.api.v1.cognify.cognify  # noqa: F401
    import cognee.infrastructure.llm.extraction  # noqa: F401
    import cognee.infrastructure.loaders.core.audio_loader  # noqa: F401
    import cognee.infrastructure.loaders.core.image_loader  # noqa: F401
    import cognee.infrastructure.loaders.core.video_loader  # noqa: F401
    assert legacy_modules() == [], f"imported at module load: {legacy_modules()}"

    async def main():
        tmp = tempfile.mkdtemp()
        audio = os.path.join(tmp, "clip.mp3")
        image = os.path.join(tmp, "pic.png")
        with open(audio, "wb") as f:
            f.write(b"\\x00" * 16)
        with open(image, "wb") as f:
            f.write(b"\\x89PNG\\r\\n\\x1a\\n")
        fake_transcription = AsyncMock(return_value=SimpleNamespace(text="hi"))
        fake_completion = AsyncMock(
            return_value=SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="a picture"))]
            )
        )
        with patch("litellm.atranscription", fake_transcription), patch(
            "litellm.acompletion", fake_completion
        ):
            transcript = await LLMGateway.create_transcript(audio, response_format="verbose_json")
            assert transcript.text == "hi"
            image_response = await LLMGateway.transcribe_image(image, prompt="describe")
            assert image_response.choices[0].message.content == "a picture"

    asyncio.run(main())
    assert legacy_modules() == [], f"imported during transcription: {legacy_modules()}"
    print("OK")
    """
)


@pytest.mark.timeout(180)
def test_default_path_never_imports_legacy_framework():
    env = {
        **os.environ,
        "STRUCTURED_OUTPUT_FRAMEWORK": "litellm_native",
        "LLM_PROVIDER": "openai",
        "LLM_API_KEY": "test-key",
        "TELEMETRY_DISABLED": "1",
    }
    env.pop("LLM_MODEL", None)
    completed = subprocess.run(
        [sys.executable, "-c", _CHILD],
        env=env,
        capture_output=True,
        text=True,
        timeout=170,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]
    assert completed.stdout.strip().endswith("OK"), completed.stdout[-2000:]
