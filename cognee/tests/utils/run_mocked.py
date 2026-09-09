"""Run an example script with every LLM and embedding call mocked.

Usage: ``uv run python cognee/tests/utils/run_mocked.py <script.py> [args...]``

Installs :func:`cognee.tests.utils.mock_ingestion.install_mocks` with an EMPTY
replay map before executing the target script, so:

* every ``LLMGateway.acreate_structured_output`` call returns its response
  model's default instance (an empty ``KnowledgeGraph``, a canned summary, an
  empty string for plain completions) with zero network calls;
* embeddings go through cognee's built-in ``MOCK_EMBEDDING`` switch — the real
  engine and tokenizer still shape chunking, but ``embed_text`` returns
  zero-vectors.

This is for the zero-assert example jobs on the PR gate: their value is "the
documented example still runs against the current API", which does not depend
on model output. Anything that asserts on real model behaviour (multimedia
transcription, provider-contract smokes, evals) must NOT run under this
runner — keep those on a real key, on the gate or in the nightly.
"""

import runpy
import sys

from cognee.tests.utils.mock_ingestion import install_mocks


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: run_mocked.py <script.py> [args...]")
    install_mocks({})
    target = sys.argv[1]
    sys.argv = sys.argv[1:]
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
