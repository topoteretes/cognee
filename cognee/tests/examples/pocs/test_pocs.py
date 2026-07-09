"""Mocked tests for examples/pocs/.

The runnable POCs are the ``*_example.py`` scripts; each runs via ``_run()``
under isolated_example_env (mocked LLM + embeddings, per-test tmp_path). The
sibling non-example modules are libraries the examples import; they have no
entrypoint, so they get an import smoke test.

Part of #3601, on the harness from #3958.
"""

from __future__ import annotations

import pytest

from cognee.tests.utils.example_runner import import_example

pytestmark = pytest.mark.asyncio


# Runnable POCs (entrypoint _run()).


@pytest.mark.xfail(
    reason=(
        "Example is broken against current cognee: its custom task "
        "extract_graph_from_data_with_entity_disambiguation_task() is called by the "
        "pipeline without the 'context' arg it requires (TypeError, independent of "
        "mocking). The example needs updating (the mocked test caught the rot)."
    ),
    strict=False,
)
async def test_disambiguate_entities_example(isolated_example_env, monkeypatch):
    # Also reads prompts/prompt1.txt via a CWD-relative path, so run from the script dir.
    from cognee.tests.utils.example_runner import EXAMPLES_ROOT

    monkeypatch.chdir(EXAMPLES_ROOT / "pocs" / "disambiguation")
    module = import_example("examples/pocs/disambiguation/disambiguate_entities_example.py")
    await module._run()


@pytest.mark.skip(reason="Requires the optional pandas dependency; not in the base keyless env.")
async def test_post_extraction_canonicalization_example(isolated_example_env):
    module = import_example(
        "examples/pocs/post_extraction_canonicalization/post_extraction_canonicalization_example.py"
    )
    await module._run()


@pytest.mark.skip(reason="Requires the optional nltk dependency; not in the base keyless env.")
async def test_prefetch_disambiguation_example(isolated_example_env):
    module = import_example(
        "examples/pocs/prefetch_disambiguation/prefetch_disambiguation_example.py"
    )
    await module._run()


# Library modules the examples import: no entrypoint, so smoke-test the import.


def test_disambiguate_entities_imports(isolated_example_env):
    import_example("examples/pocs/disambiguation/disambiguate_entities.py")


def test_extract_graph_with_entity_disambiguation_imports(isolated_example_env):
    import_example(
        "examples/pocs/disambiguation/extract_graph_from_data_with_entity_disambiguation.py"
    )


@pytest.mark.skip(reason="Requires the optional pandas dependency; not in the base keyless env.")
def test_post_extraction_canonicalization_imports(isolated_example_env):
    import_example(
        "examples/pocs/post_extraction_canonicalization/post_extraction_canonicalization.py"
    )


@pytest.mark.skip(reason="Requires the optional pandas dependency; not in the base keyless env.")
def test_prefetch_disambiguation_imports(isolated_example_env):
    import_example("examples/pocs/prefetch_disambiguation/prefetch_disambiguation.py")
