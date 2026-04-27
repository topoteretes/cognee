"""Agent loop for the shared-memory cross-domain demo.

One ``run_agent`` call executes a single agent end-to-end:

1. (SHARED only) query the shared graph for structurally-relevant prior
   findings.
2. Read own corpus.
3. Derive 3-5 findings, each grounded in corpus/memory source IDs and
   tagged from the fixed vocabulary.
4. Persist findings as StructuralFinding nodes in the target dataset,
   stamped with source_node_set="agent_{i}" via cognee's node_set.
5. Produce the final model answer, also citation-required.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

import cognee
from cognee.infrastructure.llm.LLMGateway import LLMGateway
from cognee.modules.search.types import SearchType

from data_models import StructuralFinding
from vocabulary import STRUCTURAL_TAGS, format_for_prompt


class Arm(str, Enum):
    SHARED = "shared"
    ISOLATED = "isolated"
    CONCAT = "concat"


class Finding(BaseModel):
    description: str = Field(
        description="A single structural observation about the process."
    )
    structural_tags: list[str] = Field(
        description=(
            "Tags drawn ONLY from the provided vocabulary. Each tag must be "
            "supported by the cited source(s)."
        ),
    )
    citations: list[str] = Field(
        min_length=1,
        description=(
            "At least one source ID per finding. IDs are either corpus "
            "snippet IDs (e.g., 'particle_02') or memory node IDs shown "
            "alongside retrieved memory excerpts."
        ),
    )


class FindingSet(BaseModel):
    findings: list[Finding] = Field(min_length=3, max_length=5)


class FinalAnswer(BaseModel):
    answer: str = Field(description="Final model answer synthesizing the findings.")
    citations: list[str] = Field(
        min_length=1,
        description="At least one source ID supporting the answer.",
    )


class AgentRun(BaseModel):
    agent_index: int
    arm: str
    dataset_name: str
    findings: list[Finding]
    final_answer: FinalAnswer
    n_agent_llm_calls: int
    prior_context: str


_SYSTEM_PREFIX = (
    "You are an analyst producing structural observations from field notes.\n\n"
    "STRICT CONSTRAINTS:\n"
    "- You may ONLY assert facts grounded in (a) your corpus snippets or "
    "(b) the prior context shown to you. Do NOT use background knowledge.\n"
    "- EVERY finding and answer claim MUST cite at least one CORPUS ID "
    "from your corpus (bracketed strings like '[particle_01]'). Uncited "
    "claims will be rejected.\n"
)
_SYSTEM_SUFFIX = (
    "- Tag findings using ONLY this fixed vocabulary (exact strings), and "
    "only when the cited source(s) support the tag:\n"
    f"  {format_for_prompt()}\n"
    "Do not invent tags outside this list."
)
_SYSTEM_SHARED = (
    "- A 'memory' section may be shown as lines of the form '[mem:<n>] <text>'. "
    "If a memory passage supports a structural property you are asserting "
    "from your corpus, ALSO cite that memory ID in the citations list "
    "(include BOTH a corpus ID and the mem ID). This records cross-domain "
    "agreement.\n"
)
_SYSTEM_CONCAT = (
    "- A 'prior findings' section shows earlier agents' raw findings as "
    "context. Do NOT invent citation IDs for this text — your citations "
    "list may contain ONLY corpus IDs from your own corpus. Use the prior "
    "findings to inform your analysis, but cite only your corpus.\n"
)
_SYSTEM_ISOLATED = ""


def _agent_system(arm: Arm) -> str:
    per_arm = {
        Arm.SHARED: _SYSTEM_SHARED,
        Arm.CONCAT: _SYSTEM_CONCAT,
        Arm.ISOLATED: _SYSTEM_ISOLATED,
    }[arm]
    return _SYSTEM_PREFIX + per_arm + _SYSTEM_SUFFIX


COGNIFY_PROMPT = (
    "Extract each 'Finding N' block from the text as a StructuralFinding "
    "node with:\n"
    "- description: the prose body of the finding\n"
    "- structural_tags: the listed tags (comma-separated strings)\n"
    "- citations: the listed source IDs (comma-separated strings)\n"
    "Do NOT paraphrase or merge findings."
)


def _prior_header(arm: Arm) -> str:
    if arm is Arm.SHARED:
        return "PRIOR MEMORY (cite '[mem:<n>]' IDs alongside corpus IDs where applicable):"
    if arm is Arm.CONCAT:
        return "PRIOR FINDINGS from earlier agents (context only — do NOT cite by ID):"
    return "PRIOR CONTEXT:"  # unreachable when prior_context is empty


def _build_findings_input(arm: Arm, corpus_text: str, prior_context: str) -> str:
    parts: list[str] = []
    if prior_context.strip():
        parts.append(f"{_prior_header(arm)}\n{prior_context.strip()}")
    parts.append(
        "YOUR CORPUS (cite by bracketed snippet ID):\n" + corpus_text.strip()
    )
    parts.append(
        "TASK: produce 3-5 findings. Each finding must cite at least one "
        "corpus snippet ID. Use ONLY tags from the approved vocabulary."
    )
    return "\n\n".join(parts)


def _build_answer_input(
    arm: Arm, corpus_text: str, prior_context: str, findings: FindingSet
) -> str:
    rendered = "\n\n".join(
        f"Finding {i}: {f.description}\n"
        f"tags: {', '.join(f.structural_tags)}\n"
        f"citations: {', '.join(f.citations)}"
        for i, f in enumerate(findings.findings, start=1)
    )
    parts: list[str] = []
    if prior_context.strip():
        parts.append(f"{_prior_header(arm)}\n{prior_context.strip()}")
    parts.append("YOUR CORPUS:\n" + corpus_text.strip())
    parts.append("YOUR FINDINGS (derived above):\n" + rendered)
    parts.append(
        "TASK: produce the final model answer. Cite source IDs for every "
        "claim."
    )
    return "\n\n".join(parts)


def _serialize_findings_for_cognify(findings: FindingSet) -> str:
    blocks = []
    for i, f in enumerate(findings.findings, start=1):
        blocks.append(
            f"Finding {i}.\n"
            f"Description: {f.description}\n"
            f"Structural tags: {', '.join(f.structural_tags)}\n"
            f"Citations: {', '.join(f.citations)}"
        )
    return "\n\n".join(blocks)


def _validate_findings(raw: FindingSet) -> FindingSet:
    """Drop tags outside the vocabulary and findings without citations."""
    allowed = set(STRUCTURAL_TAGS)
    cleaned: list[Finding] = []
    for f in raw.findings:
        kept_tags = [t for t in f.structural_tags if t in allowed]
        if not f.citations:
            continue
        cleaned.append(
            Finding(
                description=f.description,
                structural_tags=kept_tags,
                citations=f.citations,
            )
        )
    if len(cleaned) < 3:
        raise ValueError(
            f"Only {len(cleaned)} of {len(raw.findings)} findings survived "
            "validation (need >=3 with non-empty citations)."
        )
    return FindingSet(findings=cleaned)


def _format_search_results(results: Any) -> str:
    """Extract GRAPH_COMPLETION answers and tag them with synthetic memory IDs.

    cognee.search(..., query_type=GRAPH_COMPLETION) returns a list of dicts
    shaped like {'dataset_name': ..., 'search_result': [synthesized_text]}.
    Each synthesized text is exposed to the agent as '[mem:<i>] <text>' so
    that it can be cited directly, and memory citations are distinguishable
    from corpus citations in downstream metrics.
    """
    if not results:
        return ""
    lines: list[str] = []
    idx = 0
    for r in results:
        d = r if isinstance(r, dict) else getattr(r, "model_dump", lambda: {})()
        for text in d.get("search_result") or []:
            lines.append(f"[mem:{idx}] {text}")
            idx += 1
    return "\n".join(lines)


async def _safe_search(query: str, dataset_name: str) -> str:
    """Search a possibly-empty dataset; return empty context on failure."""
    try:
        results = await cognee.search(
            query_text=query,
            query_type=SearchType.GRAPH_COMPLETION,
            datasets=[dataset_name],
        )
    except Exception:
        return ""
    return _format_search_results(results)


async def run_agent(
    *,
    agent_index: int,
    arm: Arm,
    dataset_name: str,
    corpus_path: Path,
    prior_concat_text: Optional[str] = None,
) -> AgentRun:
    n_calls = 0
    prior_context = ""

    if arm is Arm.SHARED:
        prior_context = await _safe_search(
            query=(
                "Prior findings about a process whose observations are "
                "characterized by properties drawn from this vocabulary: "
                + format_for_prompt()
            ),
            dataset_name=dataset_name,
        )
        n_calls += 1
    elif arm is Arm.CONCAT and prior_concat_text:
        prior_context = prior_concat_text

    corpus_text = corpus_path.read_text()
    raw: FindingSet = await LLMGateway.acreate_structured_output(
        text_input=_build_findings_input(arm, corpus_text, prior_context),
        system_prompt=_agent_system(arm),
        response_model=FindingSet,
    )
    n_calls += 1
    findings = _validate_findings(raw)

    await cognee.add(
        _serialize_findings_for_cognify(findings),
        dataset_name=dataset_name,
        node_set=[f"agent_{agent_index}"],
    )
    await cognee.cognify(
        datasets=[dataset_name],
        graph_model=StructuralFinding,
        custom_prompt=COGNIFY_PROMPT,
    )

    final: FinalAnswer = await LLMGateway.acreate_structured_output(
        text_input=_build_answer_input(arm, corpus_text, prior_context, findings),
        system_prompt=_agent_system(arm),
        response_model=FinalAnswer,
    )
    n_calls += 1

    return AgentRun(
        agent_index=agent_index,
        arm=arm.value,
        dataset_name=dataset_name,
        findings=findings.findings,
        final_answer=final,
        n_agent_llm_calls=n_calls,
        prior_context=prior_context,
    )
