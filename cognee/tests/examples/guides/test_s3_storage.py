"""Test: examples/guides/s3_storage.py

The s3_storage guide reads files from real S3 buckets, making it unsuitable
for a zero-env-var CI run.  This test validates that the module is well-formed
and that ``main()`` exists, then skips execution to avoid network calls.

If you want to run this test against a real S3 bucket, set
``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``, and the environment variable
``COGNEE_TEST_RUN_S3=1`` before running pytest.
"""

import importlib.util
import os
from pathlib import Path

import pytest


def _load_example(rel_path: str):
    path = Path(__file__).parents[4] / rel_path
    spec = importlib.util.spec_from_file_location("example_s3_storage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.asyncio
async def test_s3_storage_module_is_well_formed(mock_cognee_llm, mock_cognee_embeddings):
    """s3_storage.py must be importable and expose a main() coroutine."""
    module = _load_example("examples/guides/s3_storage.py")
    assert hasattr(module, "main"), "s3_storage.py must expose a main() coroutine"

    if not os.getenv("COGNEE_TEST_RUN_S3"):
        pytest.skip(
            "S3 tests require real AWS credentials. "
            "Set COGNEE_TEST_RUN_S3=1 to enable."
        )

    await module.main()
