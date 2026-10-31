"""Ask the TrustRAG financial knowledge base with factual verification and trust gating.

Retrieves evidence from Cognee, evaluates atomic claims with cross-encoder NLI,
applies deterministic fact-checks, executes bounded corrective retrieval if needed,
and outputs a citation-grounded answer accompanied by an audit breakdown.

Run alone: uv run python examples/cookbooks/trustrag/scripts/ask.py "What was Acme's revenue in 2024?"
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

os.environ.setdefault("LOG_LEVEL", "ERROR")

COOKBOOK_DIR = Path(__file__).parent.parent
REPO_ROOT = COOKBOOK_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(COOKBOOK_DIR))

try:
    from examples.cookbooks.trustrag.config import default_config
    from examples.cookbooks.trustrag.core.answer import ask_verified_async
    from examples.cookbooks.trustrag.core.trust_score import TrustGateDecision
except ImportError:
    from config import default_config
    from core.answer import ask_verified_async
    from core.trust_score import TrustGateDecision


async def ask(question: str, dataset_name: str | None = None, verbose: bool = False) -> None:
    ds = dataset_name or default_config.dataset_name
    print(f"\n[ask] Query: {question}")
    print(f"[ask] Retrieving from dataset '{ds}' and verifying claims...")

    answer = await ask_verified_async(query=question, dataset=ds)

    print("\n" + "=" * 70)
    print("VERIFIED ANSWER:")
    print("=" * 70)
    print(answer.final_verified_answer)
    print("=" * 70)

    print(f"\nOperational Gate Decision: {answer.gate_decision.value.upper()}")
    print(f"Composite Trust Score:     {answer.trust_score:.4f} (Accept >= {default_config.trust_accept_threshold:.2f}, Escalate < {default_config.trust_escalate_threshold:.2f})")
    print(f"Total Evaluated Claims:    {answer.trust_report.total_claims}")
    print(f"  - Verified (Entailed):   {answer.trust_report.verified_claims_count}")
    print(f"  - Contradicted:          {answer.trust_report.contradicted_claims_count}")
    print(f"  - Neutral / Unproven:    {answer.trust_report.neutral_claims_count}")
    print(f"  - Unsupported / Uncited: {answer.trust_report.unsupported_claims_count}")

    if answer.corrective_result:
        cr = answer.corrective_result
        print(f"\nCorrective RAG Executed:   {cr.hops_executed} hop(s)")
        print(f"  - Resolved Claims:       {cr.claims_resolved_count}")
        print(f"  - Remaining Unverified:  {cr.claims_remaining_unverified_count}")
        print(f"  - Novel Chunks Fetched:  {cr.novel_chunks_retrieved_count}")

    if answer.trust_report.flagged_claims:
        print("\nFlagged Claims Audit:")
        for fc in answer.trust_report.flagged_claims:
            print(f"  [{fc.verdict.value.upper()}] \"{fc.claim_text}\"")
            if fc.rationale:
                print(f"    Reason: {fc.rationale}")

    if verbose and answer.retrieved_evidence:
        print("\nRetrieved Evidence Chunks:")
        for ev in answer.retrieved_evidence:
            print(f"  - [{ev.chunk_id}] (doc: {ev.doc_name}, score: {ev.score:.3f}): {ev.text[:120]}...")

    if answer.gate_decision == TrustGateDecision.ESCALATE_HUMAN:
        print(f"\n[WARNING] Answer was blocked or escalated to human auditor: {answer.trust_report.escalation_reason}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("question", nargs="+", help="financial question to answer")
    parser.add_argument("--dataset", help="Cognee dataset name to search")
    parser.add_argument("--verbose", "-v", action="store_true", help="show retrieved evidence details")
    args = parser.parse_args()

    q = " ".join(args.question)
    asyncio.run(ask(question=q, dataset_name=args.dataset, verbose=args.verbose))
