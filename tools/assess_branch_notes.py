#!/usr/bin/env python3
"""
Assess whether generated dev notes imply a documentation update is needed.

Matches the LLM integration style used by tools/generate_release_notes.py:
- uses litellm + instructor directly
- reads LLM_API_KEY / LLM_MODEL from the environment
- raises on missing dependencies, missing credentials, or LLM failures
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any


def format_markdown(assessment: Any) -> str:
    needs_update = (
        assessment.needs_documentation_update
        if hasattr(assessment, "needs_documentation_update")
        else assessment.get("needs_documentation_update")
    )
    reason = assessment.reason if hasattr(assessment, "reason") else assessment.get("reason", "")
    candidate_areas = (
        assessment.candidate_areas
        if hasattr(assessment, "candidate_areas")
        else assessment.get("candidate_areas", [])
    )
    next_steps = (
        assessment.recommended_next_steps
        if hasattr(assessment, "recommended_next_steps")
        else assessment.get("recommended_next_steps", [])
    )
    confidence = (
        assessment.confidence
        if hasattr(assessment, "confidence")
        else assessment.get("confidence", "")
    )

    lines = [
        "# Documentation Assessment",
        "",
        "## Needs documentation update",
        str(bool(needs_update)).lower(),
        "",
        "## Reason",
        reason,
        "",
        "## Candidate areas",
    ]
    lines.extend(candidate_areas or [])
    lines.extend(["", "## Recommended next steps"])
    lines.extend(next_steps or [])
    lines.extend(["", "## Confidence", confidence, ""])
    return "\n".join(lines)


async def assess_with_llm(notes_json: str, notes_markdown: str) -> Any:
    try:
        import instructor
        import litellm
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError(f"Required dependencies not available: {exc}") from exc

    api_key = os.environ.get("LLM_API_KEY")
    model = os.environ.get("LLM_MODEL", "openai/gpt-4o-mini")

    if not api_key:
        raise RuntimeError("LLM_API_KEY not set")

    class DocsAssessment(BaseModel):
        needs_documentation_update: bool = Field(
            description="Whether docs should likely be updated"
        )
        reason: str = Field(description="Why a docs update is or is not needed")
        candidate_areas: list[str] = Field(description="Likely docs areas/pages affected")
        recommended_next_steps: list[str] = Field(description="Practical next steps for docs work")
        confidence: str = Field(description="Confidence level and short explanation")

    system_prompt = """You are a documentation strategist for the Cognee project.

Analyze the provided daily dev notes and decide whether they imply documentation updates.

Guidelines:
- Focus on user-facing changes, APIs, integrations, setup, and operational behavior
- Recommend docs work only when the notes indicate likely user impact
- Keep recommendations concrete and concise
"""

    user_prompt = (
        "Determine whether the daily dev notes imply that documentation updates are needed.\n\n"
        f"Dev notes JSON:\n{notes_json}\n\n"
        f"Dev notes markdown:\n{notes_markdown}\n"
    )

    try:
        client = instructor.from_litellm(litellm.acompletion)
        return await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_model=DocsAssessment,
            api_key=api_key,
            max_retries=2,
        )
    except Exception as exc:
        raise RuntimeError(f"LLM assessment failed: {exc}") from exc


def parse_args():
    parser = argparse.ArgumentParser(description="Assess dev notes for documentation impact")
    parser.add_argument("--notes-json", required=True, type=Path)
    parser.add_argument("--notes-markdown", required=True, type=Path)
    parser.add_argument("--json-output", required=True, type=Path)
    parser.add_argument("--markdown-output", required=True, type=Path)
    return parser.parse_args()


async def main():
    args = parse_args()
    notes_json = args.notes_json.read_text()
    notes_markdown = args.notes_markdown.read_text()

    assessment = await assess_with_llm(notes_json, notes_markdown)

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(
            assessment.model_dump() if hasattr(assessment, "model_dump") else assessment,
            indent=2,
        )
        + "\n"
    )
    args.markdown_output.write_text(format_markdown(assessment))
    print(args.markdown_output.read_text())


if __name__ == "__main__":
    asyncio.run(main())
