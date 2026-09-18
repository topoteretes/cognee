"""Run an example script with every LLM and embedding call mocked.

Usage: ``uv run python cognee/tests/utils/run_mocked.py [--replay <map.json>] <script.py> [args...]``

Installs :func:`cognee.tests.utils.mock_ingestion.install_mocks` before executing
the target script. By default the replay map is EMPTY, so:

* every ``LLMGateway.acreate_structured_output`` call returns its response
  model's default instance (an empty ``KnowledgeGraph``, a canned summary, an
  empty string for plain completions) with zero network calls;
* embeddings go through cognee's built-in ``MOCK_EMBEDDING`` switch — the real
  engine and tokenizer still shape chunking, but ``embed_text`` returns
  zero-vectors.

``--replay <map.json>`` runs the script in seeded mode instead. The JSON file
is the ``{title: {"knowledge_graph": ..., "summary": ...}}`` map
``install_mocks`` takes, matched by substring of the extraction prompt, so the
graph gets real entities and edges; and embeddings become deterministic
pseudo-random unit vectors derived from the text instead of zeros, because a
zero query vector has no cosine similarity to anything and every vector search
would still come back empty. Use it for a script whose documented path needs a
populated, searchable graph: a graph completion on an entity-less graph has
nothing to retrieve and correctly returns no answer, which such a script cannot
demonstrate. Completions are still the model's default (an empty string), so
seeded mode asserts nothing about model output either. The seed is per
invocation; every other mocked run stays on the empty map and zero vectors.

This is for the zero-assert example jobs on the PR gate: their value is "the
documented example still runs against the current API", which does not depend
on model output. Anything that asserts on real model behaviour (multimedia
transcription, provider-contract smokes, evals) must NOT run under this
runner — keep those on a real key, on the gate or in the nightly.
"""

import hashlib
import json
import math
import random
import runpy
import sys
from pathlib import Path

from cognee.tests.utils.mock_ingestion import install_mocks

USAGE = "usage: run_mocked.py [--replay <map.json>] <script.py> [args...]"


def _hashed_unit_vector(text: str, dimensions: int) -> list[float]:
    """A deterministic pseudo-random unit vector for ``text``.

    Same text, same vector, on every run and platform; different texts,
    (almost surely) different directions. Nothing about it is semantic: it only
    makes cosine similarity defined so vector search returns *something*.
    """
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    rng = random.Random(seed)
    vector = [rng.gauss(0.0, 1.0) for _ in range(dimensions)]
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _install_hashed_embeddings() -> None:
    """Replace ``embed_text`` on the configured engine class with hashed vectors.

    Class-level, like the benchmark's document-embedding replay, so instances
    constructed later (caches are cleared by ``install_mocks``) use it too. The
    real engine is still constructed: its tokenizer keeps deciding chunk sizes.
    """
    from cognee.infrastructure.databases.vector.embeddings.get_embedding_engine import (
        get_embedding_engine,
    )

    engine_cls = type(get_embedding_engine())

    async def hashed_embed(self, text):
        return [_hashed_unit_vector(item, self.dimensions) for item in text]

    engine_cls.embed_text = hashed_embed


def main() -> None:
    args = sys.argv[1:]
    replay_file: Path | None = None
    if args and args[0] == "--replay":
        if len(args) < 2:
            sys.exit(USAGE)
        replay_file = Path(args[1])
        args = args[2:]
    if not args:
        sys.exit(USAGE)

    if replay_file is None:
        install_mocks({})
    else:
        install_mocks(json.loads(replay_file.read_text()), mock_embeddings=False)
        _install_hashed_embeddings()

    target = args[0]
    sys.argv = args
    runpy.run_path(target, run_name="__main__")


if __name__ == "__main__":
    main()
