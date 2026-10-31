"""Keyless end to end: every stored chunk fits the embedding model's window (SDK-810).

Chunk sizing counts tokens with the embedding model's own tokenizer and cuts at
``resolve_chunk_size()``. Two things can silently truncate what gets embedded: a
budget larger than the model's window (fastembed truncates without a word; SDK-868
lowers the budget to the model's input limit), and a tokenizer that mis-counts (a
``tokenizer.json`` with stored truncation or fixed padding makes every text count
as 128 tokens, or caps long ones; SDK-810 switches them off). This check
ingests a long text through the default pipeline and then verifies, chunk by
chunk, against two oracles: ``transformers.AutoTokenizer`` (independent counts;
installed with the GLiNER runtime) and fastembed's own tokenizer (what the ONNX
model actually sees, truncation enabled at its window).

Run: ``python cognee/tests/e2e/keyless/chunk_token_budget_check.py [fastembed model]``

With no argument the keyless default (BAAI/bge-small-en-v1.5) is used. The CI job
also runs it with ``snowflake/snowflake-arctic-embed-xs``, whose tokenizer.json
stores truncation at 512 (a counter that honoured it would report at most 512
tokens for any document), and with ``sentence-transformers/all-MiniLM-L6-v2``,
whose tokenizer.json stores truncation and a fixed padding length of 128 (every
text would count as 128 tokens, and fastembed, left with that padding, fails on
a batch mixing texts above and below it).
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

for key in list(os.environ):
    if key.startswith(("LLM_", "EMBEDDING_", "OPENAI_", "GRAPH_EXTRACTOR", "BAML_")):
        os.environ.pop(key)
for key in ("COGNEE_SKIP_PREFLIGHT", "COGNEE_SKIP_CONNECTION_TEST", "MOCK_EMBEDDING"):
    os.environ.pop(key, None)
if len(sys.argv) > 1:
    os.environ["EMBEDDING_PROVIDER"] = "fastembed"
    os.environ["EMBEDDING_MODEL"] = sys.argv[1]

ROOT = Path(__file__).resolve().parent / ".chunk_budget_run"
ROOT.mkdir(exist_ok=True)
os.chdir(ROOT)  # pydantic-settings reads a cwd-relative .env; there is none here
os.environ["DATA_ROOT_DIRECTORY"] = str(ROOT / "data")
os.environ["SYSTEM_ROOT_DIRECTORY"] = str(ROOT / "system")
os.environ["TELEMETRY_DISABLED"] = "1"

import dotenv  # noqa: E402

dotenv.load_dotenv = lambda *args, **kwargs: False  # cognee/__init__.py would load the repo .env

import cognee  # noqa: E402

TEXT_PATH = Path(__file__).resolve().parents[2] / "test_data" / "alice_in_wonderland.txt"
WORDS = 8000  # ~10k tokens: dozens of chunks at a 512 budget, without a long GLiNER run

logging.getLogger().setLevel(logging.WARNING)


async def main() -> None:
    from cognee.infrastructure.databases.graph import get_graph_engine
    from cognee.infrastructure.databases.vector.embeddings.get_embedding_engine import (
        get_embedding_engine,
    )
    from cognee.infrastructure.llm.utils import resolve_chunk_size

    text = " ".join(TEXT_PATH.read_text(encoding="utf-8").split()[:WORDS])
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)
    await cognee.add(text, dataset_name="chunk_budget")
    # The embedding tokenizer must not import transformers: it caches "no torch" at
    # import, and cognify's GLiNER auto-installer adds torch only after add() (#5258).
    assert "transformers" not in sys.modules, (
        "transformers was imported before cognify; the GLiNER runtime install would then "
        "fail with 'PyTorch not found' in this process"
    )
    await cognee.cognify(datasets=["chunk_budget"])

    engine = get_embedding_engine()
    budget = await resolve_chunk_size(None)  # what cognify sized the chunks by
    # What the ONNX model sees: fastembed's tokenizer, truncation enabled at the window.
    model_tokenizer = engine.embedding_model.model.tokenizer
    window = model_tokenizer.truncation["max_length"]
    specials = model_tokenizer.num_special_tokens_to_add(is_pair=False)
    print(f"model={engine.model} window={window} specials={specials} chunk_budget={budget}")
    problems = []
    if budget + specials > window:
        problems.append(
            f"chunk budget {budget} + {specials} special tokens exceeds the model window "
            f"{window}: fastembed would truncate full chunks"
        )

    # Independent oracle for counts (installed with the GLiNER runtime, imported after torch).
    from transformers import AutoTokenizer

    oracle = AutoTokenizer.from_pretrained(engine.model)
    # Whole-document counting, as the --dry-run estimate and whole-chunk counters use it:
    # a tokenizer.json with stored truncation would cap this at the stored length.
    total = len(oracle.tokenize(text))
    counted = engine.tokenizer.count_tokens(text)
    if counted != total:
        problems.append(
            f"the engine tokenizer counts the document as {counted} tokens, not {total}"
        )

    nodes, _ = await (await get_graph_engine()).get_graph_data()
    chunks = [props for _, props in nodes if props.get("type") == "DocumentChunk"]
    chunk_ids = {str(node_id) for node_id, props in nodes if props.get("type") == "DocumentChunk"}
    if len(chunks) < 5:
        problems.append(f"expected a multi-chunk document, got {len(chunks)} chunks")

    for chunk in sorted(chunks, key=lambda c: c["chunk_index"]):
        text_ = chunk["text"]
        truth = len(oracle.tokenize(text_))
        believed = chunk["chunk_size"]
        if chunk.get("max_chunk_tokens") != budget:
            problems.append(
                f"chunk {chunk['chunk_index']}: max_chunk_tokens={chunk.get('max_chunk_tokens')} != {budget}"
            )
        if truth > budget:
            problems.append(f"chunk {chunk['chunk_index']}: {truth} tokens > budget {budget}")
        if model_tokenizer.encode(text_).overflowing:
            problems.append(
                f"chunk {chunk['chunk_index']}: fastembed truncates it ({truth} tokens, window {window})"
            )
        if abs(believed - truth) > max(2, truth // 20):
            problems.append(
                f"chunk {chunk['chunk_index']}: chunker counted {believed}, tokenizer says {truth}"
            )

    # Every chunk must have been embedded: its id must come back from the vector store.
    from cognee.infrastructure.databases.vector import get_vector_engine_async

    vector_engine = await get_vector_engine_async()
    hits = await vector_engine.search(
        collection_name="DocumentChunk_text", query_text="Alice", limit=len(chunks) + 5
    )
    missing = chunk_ids - {str(hit.id) for hit in hits}
    if missing:
        problems.append(f"{len(missing)} of {len(chunks)} chunks have no stored embedding")
    assert not problems, "\n".join(problems)

    stored = sum(len(oracle.tokenize(chunk["text"])) for chunk in chunks)
    assert abs(stored - total) <= total // 50, f"chunks hold {stored} tokens, the document {total}"
    print(
        f"{len(chunks)} chunks, {stored} tokens stored of {total}, largest "
        f"{max(len(oracle.tokenize(c['text'])) for c in chunks)} <= budget {budget}"
    )
    print("chunk token budget: PASS")


if __name__ == "__main__":
    asyncio.run(main())
