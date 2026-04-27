"""Orchestrator for the shared-memory cross-domain demo.

Runs N seeds of one experimental arm. Each seed: prune cognee state, run
three agents sequentially (with arm-appropriate dataset wiring), score
each agent's final answer with the rubric, write ``results/{arm}_s{seed}.json``.

Usage
-----

::

    python run_demo.py --arm shared   --seeds 5
    python run_demo.py --arm isolated --seeds 5
    python run_demo.py --arm concat   --seeds 5
    python run_demo.py --arm shared   --seeds 5 --start-seed 5  # seeds 5..9

Runtime prerequisites
---------------------

Loads ``/Users/veljko/coding/cognee/.env`` (override with
``COGNEE_ENV_PATH``) and sets sane defaults for the current env's
embedding model: ``EMBEDDING_DIMENSIONS=1536`` (text-embedding-3-small)
and ``COGNEE_SKIP_CONNECTION_TEST=true`` (cognee's 30 s probe is too
tight for this OpenAI endpoint). State is stored under
``$SHARED_MEM_DEMO_STATE`` (default ``~/.cache/shared_mem_demo``), not
the user's global cognee dir.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

DEMO_DIR = Path(__file__).resolve().parent

_ENV_PATH = Path(os.environ.get("COGNEE_ENV_PATH", "/Users/veljko/coding/cognee/.env"))
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)
os.environ.setdefault("COGNEE_SKIP_CONNECTION_TEST", "true")
os.environ.setdefault("EMBEDDING_DIMENSIONS", "1536")
# Session memory registers a SQLAlchemy table that is not idempotent
# across prune cycles; disabling it avoids "Table 'session_records' is
# already defined" on the second seed of any arm.
os.environ.setdefault("CACHING", "false")

sys.path.insert(0, str(DEMO_DIR))

import cognee  # noqa: E402

_STATE = Path(
    os.environ.get("SHARED_MEM_DEMO_STATE", Path.home() / ".cache" / "shared_mem_demo")
)
cognee.config.data_root_directory(str(_STATE / "data"))
cognee.config.system_root_directory(str(_STATE / "system"))

from agents import AgentRun, Arm, run_agent  # noqa: E402
from rubric import score_abstraction, score_correctness  # noqa: E402


CORPORA: dict[int, Path] = {
    1: DEMO_DIR / "corpora" / "agent_1_particle.md",
    2: DEMO_DIR / "corpora" / "agent_2_material.md",
    3: DEMO_DIR / "corpora" / "agent_3_market.md",
}
RESULTS_DIR = DEMO_DIR / "results"


def _dataset_name(arm: Arm, seed: int, agent_index: int) -> str:
    if arm is Arm.SHARED:
        return f"shared_s{seed}"
    return f"{arm.value}_a{agent_index}_s{seed}"


def _count_memory_citations(run: AgentRun) -> tuple[int, int]:
    in_findings = sum(
        1 for f in run.findings for c in f.citations if c.startswith("mem:")
    )
    in_final = sum(1 for c in run.final_answer.citations if c.startswith("mem:"))
    return in_findings, in_final


def _render_concat(prior_runs: list[AgentRun]) -> str:
    """Serialize earlier agents' findings as raw text for the CONCAT arm.

    Deliberately no bracketed IDs in the rendered text — the CONCAT agent
    must not cite this prior text by ID (the arm-specific system prompt
    says so explicitly), and we avoid any tokens that might look like a
    citation label and be hallucinated back as a citation.
    """
    blocks: list[str] = []
    for pr in prior_runs:
        lines = [f"From agent_{pr.agent_index}:"]
        for f in pr.findings:
            tags = ", ".join(f.structural_tags) if f.structural_tags else ""
            lines.append(f"- tags=({tags}): {f.description}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


async def run_seed(arm: Arm, seed: int) -> dict:
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    agent_runs: list[AgentRun] = []
    for i in (1, 2, 3):
        prior_concat = _render_concat(agent_runs) if arm is Arm.CONCAT and i > 1 else None
        run = await run_agent(
            agent_index=i,
            arm=arm,
            dataset_name=_dataset_name(arm, seed, i),
            corpus_path=CORPORA[i],
            prior_concat_text=prior_concat,
        )
        agent_runs.append(run)

    agents_out: list[dict] = []
    for run in agent_runs:
        corr = await score_correctness(run.final_answer.answer)
        mem_f, mem_a = _count_memory_citations(run)
        agents_out.append(
            {
                "agent_index": run.agent_index,
                "n_llm_calls": run.n_agent_llm_calls,
                "n_findings": len(run.findings),
                "findings": [f.model_dump() for f in run.findings],
                "final_answer": run.final_answer.answer,
                "final_answer_citations": run.final_answer.citations,
                "correctness": corr.score(),
                "correctness_breakdown": [c.model_dump() for c in corr.checks],
                "memory_citations_in_findings": mem_f,
                "memory_citations_in_final": mem_a,
                "prior_context_chars": len(run.prior_context),
            }
        )

    abs_res = await score_abstraction(agent_runs[-1].final_answer.answer)

    return {
        "arm": arm.value,
        "seed": seed,
        "agents": agents_out,
        "a3_unifies_domains": abs_res.unifies_domains,
        "a3_abstraction_justification": abs_res.one_line_justification,
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=[a.value for a in Arm], required=True)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--start-seed", type=int, default=0)
    args = parser.parse_args()

    arm = Arm(args.arm)
    RESULTS_DIR.mkdir(exist_ok=True)

    for seed in range(args.start_seed, args.start_seed + args.seeds):
        print(f"\n=== {arm.value} seed={seed} ===", flush=True)
        result = await run_seed(arm, seed)
        out_path = RESULTS_DIR / f"{arm.value}_s{seed}.json"
        out_path.write_text(json.dumps(result, indent=2))
        a = result["agents"]
        print(
            f"  wrote {out_path.name}  "
            f"correctness A1={a[0]['correctness']}/8 "
            f"A2={a[1]['correctness']}/8 "
            f"A3={a[2]['correctness']}/8  "
            f"A3-unifies={result['a3_unifies_domains']}  "
            f"mem-cites A2f={a[1]['memory_citations_in_findings']} "
            f"A3f={a[2]['memory_citations_in_findings']}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
