"""brainbox task: code repository -> architecture graph, promoted into the brain.

Runs INSIDE the sandbox. Three moves, each against a different wall:

  1. recall  — ask the brain a question with the scoped identity (only the
               granted datasets answer; the brain does the LLM work).
  2. build   — `cognee.remember(repo, content_type="code")` locally: the enola
               code graph, deterministic, no LLM, no embeddings, no network.
               Then the module-level `architecture` view with a Mermaid diagram.
  3. promote — `cognee.push(...)`: export the local graph as a COGX archive and
               import it into the brain, zero LLM calls. The target dataset is
               owned by the sandbox identity; the human promotes from there.

Environment (set by brainbox/run.sh):
  COGNEE_SERVICE_URL   brain URL as seen from the sandbox (host.docker.internal:<port>)
  COGNEE_API_KEY       the sbx placeholder — the proxy swaps in the real key
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "true")
os.environ.setdefault("COGNEE_SKIP_CONNECTION_TEST", "true")

import _compat  # proxy-awareness for cognee releases without PR #5196; side-effect import

import cognee
from cognee.cli.code_search import write_diagram

assert _compat  # keep the side-effect import from being pruned

BRAIN = os.environ["COGNEE_SERVICE_URL"].rstrip("/")
KEY = os.environ["COGNEE_API_KEY"]


def brain_post(path: str, body: dict, timeout: float = 300.0) -> tuple[int, dict | list | str]:
    request = urllib.request.Request(
        BRAIN + path,
        method="POST",
        data=json.dumps(body).encode(),
        headers={"X-Api-Key": KEY, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode(errors="replace")


def step(title: str) -> None:
    print(f"\n== {title}", flush=True)


async def main(args: argparse.Namespace) -> None:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"brain": BRAIN, "repo": args.repo}

    step("1/3 recall from the brain (scoped identity)")
    t = time.time()
    status, body = brain_post(
        "/api/v1/recall", {"query": args.question, "searchType": "GRAPH_COMPLETION", "topK": 5}
    )
    items = body if isinstance(body, list) else []
    answers = [
        {"dataset": item.get("dataset_name"), "text": item.get("text")}
        for item in items
        if isinstance(item, dict)
    ]
    report["recall"] = {"http": status, "seconds": round(time.time() - t, 1), "answers": answers}
    (out / "recall.json").write_text(json.dumps(report["recall"], indent=2))
    for answer in answers:
        print(f"  [{answer['dataset']}] {str(answer['text'])[:200]}")
    if not answers:
        print(f"  HTTP {status}: {str(body)[:200]}")

    step("2/3 build the code graph locally (no LLM, no network)")
    t = time.time()
    result = await cognee.remember(
        args.repo, dataset_name=args.dataset, content_type="code", self_improvement=False
    )
    arch = await cognee.search(
        query_type=cognee.SearchType.CODE,
        query_text="",
        datasets=[args.dataset],
        code_query={"operation": "architecture", "diagram": "mermaid"},
    )
    payload = arch[0]["search_result"] if arch and "search_result" in arch[0] else arch
    stats = payload.get("stats", {}) if isinstance(payload, dict) else {}
    (out / "architecture.json").write_text(json.dumps(payload, indent=2, default=str))
    html_path = write_diagram(arch, str(out / "architecture.html"))
    mermaid = (payload.get("diagram") or {}).get("source", "") if isinstance(payload, dict) else ""
    (out / "architecture.mmd").write_text(mermaid)
    report["build"] = {
        "status": getattr(result, "status", None),
        "seconds": round(time.time() - t, 1),
        "modules": stats.get("nodes_total"),
        "module_edges": stats.get("edges_shown"),
        "diagram": html_path,
    }
    print(f"  remember: {report['build']['status']} in {report['build']['seconds']}s")
    print(f"  architecture: {stats.get('nodes_total')} modules, {stats.get('edges_shown')} edges")
    print(f"  diagram: {html_path}")

    step("3/3 promote the graph into the brain (COGX push, zero LLM calls)")
    t = time.time()
    pushed = await cognee.push(args.dataset, target_dataset=args.target, mode="preserve")
    report["promote"] = {
        "status": pushed.status,
        "target_dataset": pushed.target_dataset,
        "nodes": pushed.num_nodes,
        "edges": pushed.num_edges,
        "seconds": round(time.time() - t, 1),
    }
    print(
        f"  pushed {pushed.num_nodes} nodes / {pushed.num_edges} edges "
        f"-> '{pushed.target_dataset}' ({pushed.status}) in {report['promote']['seconds']}s"
    )

    (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nreport: {out / 'report.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="code repository path inside the sandbox")
    parser.add_argument("--dataset", default="sandbox_code", help="local scratch dataset")
    parser.add_argument("--target", required=True, help="output dataset name on the brain")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--question",
        default="What do we know about this codebase and who owns it?",
        help="what to ask the brain before building",
    )
    asyncio.run(main(parser.parse_args()))
