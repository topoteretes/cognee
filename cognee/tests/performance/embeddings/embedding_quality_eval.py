"""Compare an embedding model's retrieval against a reference model on War and Peace.

Each environment is a persistent cognee store holding the same passages of the book,
embedded with one model. ``build`` creates one; ``compare`` ranks every chunk for a set
of wedding questions in two environments and reports where known wedding passages land
and how much each candidate's top k agrees with the reference's.

Why identical passages: chunk boundaries follow the embedding model's own tokenizer, so
two models given the same book and chunk size cut different chunks, and boundary effects
then mix with model effects. The book is therefore cut once (``passages.jsonl``, with
the chunker of cognee's default local model, BAAI/bge-small-en-v1.5, at CHUNK_SIZE) and
every passage is added as its own document. The indexing pipeline stores each passage as
exactly one chunk in every environment (``build`` verifies this), so every model ranks
the same texts.

Usage (the API key for a hosted model comes from the shell, never from the env files):

    python embedding_quality_eval.py --workdir ~/embedding-envs build \\
        --name openai-3-large --provider openai \\
        --model openai/text-embedding-3-large --dimensions 3072
    python embedding_quality_eval.py --workdir ~/embedding-envs build \\
        --name fastembed-bge-small --provider fastembed \\
        --model BAAI/bge-small-en-v1.5 --dimensions 384
    python embedding_quality_eval.py --workdir ~/embedding-envs compare \\
        --reference openai-3-large --candidate fastembed-bge-small

An environment stays usable for other experiments: point cognee at it with
``COGNEE_ENV_FILE=<workdir>/<name>/.env`` and search the ``war_and_peace`` dataset.
"""

import argparse
import asyncio
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import urllib.request

BOOK_URL = "https://www.gutenberg.org/cache/epub/2600/pg2600.txt"
DATASET = "war_and_peace"
CHUNK_SIZE = 512
COLLECTION = "DocumentChunk_text"
SEGMENTER = {
    "EMBEDDING_PROVIDER": "fastembed",
    "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
    "EMBEDDING_DIMENSIONS": "384",
}

# Passages about weddings and marriages, each identified by a phrase only it contains.
# The first four are the ones the BROAD search investigation tracked (SDK-324).
PASSAGES = {
    "pierre_marries_helene": "Six weeks later he was married",
    "helene_dies": "Countess Hélène Bezúkhova had suddenly died",
    "natasha_had_married": "Natásha had married in the early spring",
    "pierre_under_wifes_thumb": "Pierre was under his wife’s thumb",
    "berg_vera_wedding": "Berg had already been engaged a month",
    "boris_julie_wedding": "prepared everything for a brilliant wedding",
    "natasha_wedding_1813": "Natásha’s wedding to Bezúkhov, which took place in 1813",
    "nicholas_marries_mary": "In the winter of 1813 Nicholas married Princess Mary",
    "anatole_was_married": "told Natásha that Anatole was married",
}

QUERIES = [
    # The seven phrasings of the BROAD investigation, verbatim.
    "How many wives did Pierre have?",
    "Pierre marriage wife",
    "Pierre married Hélène and Natasha",
    "Pierre’s first wife Hélène",
    "Pierre and Natasha married",
    "Hélène died",
    "Pierre's wedding to Hélène",
    # The wedding topic beyond Pierre.
    "wedding",
    "Who got married in War and Peace?",
    "preparations for a wedding and the bride's dowry",
    "Nicholas and Princess Mary get married",
    "Anatole is secretly already married",
]


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def text_key(text: str) -> str:
    return hashlib.sha256(normalize(text).encode()).hexdigest()[:16]


def env_dir(workdir: pathlib.Path, name: str) -> pathlib.Path:
    return workdir / "environments" / name


def env_file_content(directory: pathlib.Path, settings: dict[str, str]) -> str:
    values = {
        **settings,
        "DATA_ROOT_DIRECTORY": str(directory / "data"),
        "SYSTEM_ROOT_DIRECTORY": str(directory / "system"),
        "CACHE_ROOT_DIRECTORY": str(directory / "cache"),
        # One shared store per environment, so later experiments need no user setup.
        "ENABLE_BACKEND_ACCESS_CONTROL": "false",
        "TELEMETRY_DISABLED": "1",
    }
    return "".join(f"{key}={value}\n" for key, value in values.items())


def run_child(workdir: pathlib.Path, env_file: pathlib.Path, *args: str) -> None:
    """Run a step in its own process: cognee reads its configuration once, at import."""
    subprocess.run(
        [sys.executable, __file__, "--workdir", str(workdir), *args],
        env={**os.environ, "COGNEE_ENV_FILE": str(env_file)},
        check=True,
    )


# ---------------------------------------------------------------- passages


def ensure_passages(workdir: pathlib.Path) -> pathlib.Path:
    """Cut the book into passages once; every environment reuses the same file."""
    passages = workdir / "passages.jsonl"
    if passages.exists():
        return passages
    book = workdir / "war_and_peace.txt"
    if not book.exists():
        urllib.request.urlretrieve(BOOK_URL, book)
    segmenter = workdir / "segmenter"
    segmenter.mkdir(parents=True, exist_ok=True)
    env_file = segmenter / ".env"
    env_file.write_text(env_file_content(segmenter, SEGMENTER))
    run_child(workdir, env_file, "_segment")
    return passages


def _segment(workdir: pathlib.Path) -> None:
    from cognee.infrastructure.llm.utils import resolve_chunk_size
    from cognee.tasks.chunks import chunk_by_paragraph

    raw = (workdir / "war_and_peace.txt").read_text(encoding="utf-8")
    start = raw.index("*** START OF")
    body = raw[raw.index("\n", start) + 1 : raw.index("*** END OF")]
    max_size = asyncio.run(resolve_chunk_size(CHUNK_SIZE))
    texts = [c["text"] for c in chunk_by_paragraph(body, max_size) if c["text"].strip()]
    with (workdir / "passages.jsonl").open("w", encoding="utf-8") as out:
        for text in texts:
            out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    print(f"Cut {len(texts)} passages of at most {max_size} tokens.")


# ---------------------------------------------------------------- build


def build(workdir: pathlib.Path, args) -> None:
    directory = env_dir(workdir, args.name)
    if directory.exists():
        raise SystemExit(f"Environment {directory} already exists; pick another --name.")
    passages = ensure_passages(workdir)
    settings = {
        "EMBEDDING_PROVIDER": args.provider,
        "EMBEDDING_MODEL": args.model,
        "EMBEDDING_DIMENSIONS": str(args.dimensions),
    }
    for setting in args.setting:
        key, _, value = setting.partition("=")
        settings[key] = value
    directory.mkdir(parents=True)
    env_file = directory / ".env"
    env_file.write_text(env_file_content(directory, settings))
    run_child(workdir, env_file, "_build", "--passages", str(passages))


async def _build(passages_file: pathlib.Path) -> None:
    import cognee
    from cognee.api.v1.cognify.cognify import get_indexing_tasks
    from cognee.infrastructure.databases.vector import get_vector_engine_async
    from cognee.modules.pipelines import run_pipeline
    from cognee.modules.users.methods import get_default_user
    from cognee.shared.pipeline_names import INDEXING_PIPELINE_NAME

    texts = [json.loads(line)["text"] for line in passages_file.open(encoding="utf-8")]
    await cognee.add(texts, dataset_name=DATASET)
    async for _ in run_pipeline(
        # The passages fix the chunk boundaries; each model's own automatic chunk size
        # only has to be large enough to keep a passage whole (checked below).
        tasks=await get_indexing_tasks(),
        datasets=[DATASET],
        user=await get_default_user(),
        pipeline_name=INDEXING_PIPELINE_NAME,
        needs_llm=False,
    ):
        pass
    vector_engine = await get_vector_engine_async()
    stored = await (await vector_engine.get_collection(COLLECTION)).count_rows()
    if stored != len(texts):
        raise RuntimeError(
            f"{stored} chunks stored for {len(texts)} passages: this model's chunk size "
            "splits some passages, so its chunks would differ from the other environments'."
        )
    print(f"Indexed {stored} passages, one chunk each.")


# ---------------------------------------------------------------- rank


async def _rank(out: pathlib.Path) -> None:
    from cognee.infrastructure.databases.vector import get_vector_engine_async
    from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_config

    vector_engine = await get_vector_engine_async()
    rankings = {}
    texts = {}
    for query in QUERIES:
        results = await vector_engine.search(
            COLLECTION, query_text=query, limit=None, include_payload=True
        )
        keys = []
        for result in results:
            key = text_key(result.payload["text"])
            texts[key] = result.payload["text"]
            keys.append(key)
        rankings[query] = keys

    passage_keys = {}
    for name, anchor in PASSAGES.items():
        matches = [key for key, text in texts.items() if anchor in normalize(text)]
        if len(matches) != 1:
            raise RuntimeError(f"Anchor of {name!r} found in {len(matches)} chunks, need 1.")
        passage_keys[name] = matches[0]

    out.write_text(
        json.dumps(
            {
                "model": get_embedding_config().embedding_model,
                "chunks": len(texts),
                "passages": passage_keys,
                "rankings": rankings,
            }
        )
    )


def rank(workdir: pathlib.Path, name: str) -> dict:
    directory = env_dir(workdir, name)
    if not directory.exists():
        raise SystemExit(f"No environment {directory}; run build first.")
    out = directory / "rankings.json"
    run_child(workdir, directory / ".env", "_rank", "--out", str(out))
    return json.loads(out.read_text())


# ---------------------------------------------------------------- compare


def compare(workdir: pathlib.Path, args) -> None:
    k = args.k
    reference = rank(workdir, args.reference)
    candidate = rank(workdir, args.candidate)
    if set(reference["rankings"][QUERIES[0]]) != set(candidate["rankings"][QUERIES[0]]):
        raise SystemExit("The environments hold different chunks; rebuild from one passages file.")

    def position(run: dict, query: str, passage: str) -> int:
        return run["rankings"][query].index(run["passages"][passage]) + 1

    ranks = {
        query: {
            passage: {
                "reference": position(reference, query, passage),
                "candidate": position(candidate, query, passage),
            }
            for passage in PASSAGES
        }
        for query in QUERIES
    }
    overlap = {
        query: len(set(reference["rankings"][query][:k]) & set(candidate["rankings"][query][:k]))
        / k
        for query in QUERIES
    }
    per_passage = {
        passage: {
            side: {
                "best_rank": min(ranks[q][passage][side] for q in QUERIES),
                "queries_in_top_k": sum(ranks[q][passage][side] <= k for q in QUERIES),
            }
            for side in ("reference", "candidate")
        }
        for passage in PASSAGES
    }
    report = {
        "reference": reference["model"],
        "candidate": candidate["model"],
        "chunks": reference["chunks"],
        "k": k,
        "mean_overlap_at_k": sum(overlap.values()) / len(overlap),
        "hits_at_k": {
            side: sum(ranks[q][p][side] <= k for q in QUERIES for p in PASSAGES)
            for side in ("reference", "candidate")
        },
        "passages_found_at_k": {
            side: sum(v[side]["queries_in_top_k"] > 0 for v in per_passage.values())
            for side in ("reference", "candidate")
        },
        "per_passage": per_passage,
        "overlap_at_k": overlap,
        "ranks": ranks,
    }
    reports = workdir / "reports"
    reports.mkdir(exist_ok=True)
    path = reports / f"{args.candidate}_vs_{args.reference}_k{k}.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print_report(report, args.reference, args.candidate)
    print(f"\nFull report: {path}")


def print_report(report: dict, reference: str, candidate: str) -> None:
    k, pairs = report["k"], len(QUERIES) * len(PASSAGES)
    print(f"\nReference {reference} ({report['reference']})")
    print(f"Candidate {candidate} ({report['candidate']})")
    print(f"{report['chunks']} chunks, {len(QUERIES)} queries, {len(PASSAGES)} passages, k={k}\n")
    print("| | reference | candidate |\n|---|---|---|")
    hits, found = report["hits_at_k"], report["passages_found_at_k"]
    print(
        f"| query-passage pairs in top {k} (of {pairs}) | {hits['reference']} | {hits['candidate']} |"
    )
    print(
        f"| passages found by any query (of {len(PASSAGES)}) | {found['reference']} | {found['candidate']} |"
    )
    print(f"| mean overlap with reference top {k} | 1.00 | {report['mean_overlap_at_k']:.2f} |")
    print(f"\nBest rank of each passage over all queries (queries putting it in the top {k}):\n")
    print("| passage | reference | candidate |\n|---|---|---|")
    for passage, sides in report["per_passage"].items():
        cells = [f"{s['best_rank']} ({s['queries_in_top_k']})" for s in sides.values()]
        print(f"| {passage} | {cells[0]} | {cells[1]} |")
    print(f"\nPer query: overlap with reference top {k}, and rank reference / candidate:\n")
    print("| query | overlap | " + " | ".join(PASSAGES) + " |")
    print("|---" * (len(PASSAGES) + 2) + "|")
    for query, row in report["ranks"].items():
        cells = [f"{r['reference']} / {r['candidate']}" for r in row.values()]
        print(f"| {query} | {report['overlap_at_k'][query]:.2f} | " + " | ".join(cells) + " |")


# ---------------------------------------------------------------- cli


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workdir", type=pathlib.Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)

    build_cmd = commands.add_parser("build", help="create a persistent environment")
    build_cmd.add_argument("--name", required=True)
    build_cmd.add_argument("--provider", required=True)
    build_cmd.add_argument("--model", required=True)
    build_cmd.add_argument("--dimensions", type=int, required=True)
    build_cmd.add_argument(
        "--setting",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra cognee setting for the environment, e.g. EMBEDDING_ENDPOINT=...",
    )

    compare_cmd = commands.add_parser("compare", help="compare a candidate to the reference")
    compare_cmd.add_argument("--reference", required=True)
    compare_cmd.add_argument("--candidate", required=True)
    compare_cmd.add_argument("--k", type=int, default=30)

    commands.add_parser("_segment")
    commands.add_parser("_build").add_argument("--passages", type=pathlib.Path, required=True)
    commands.add_parser("_rank").add_argument("--out", type=pathlib.Path, required=True)

    args = parser.parse_args()
    workdir = args.workdir.expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    if args.command == "build":
        build(workdir, args)
    elif args.command == "compare":
        compare(workdir, args)
    elif args.command == "_segment":
        _segment(workdir)
    elif args.command == "_build":
        asyncio.run(_build(args.passages))
    elif args.command == "_rank":
        asyncio.run(_rank(args.out))


if __name__ == "__main__":
    main()
