"""Reference-free groundedness judge for real user questions.

Real questions collected from tenant traffic have no golden answer, so they cannot be
scored for correctness. This adapter judges only whether the answer cognee produced is
actually supported by the context that was retrieved for it.
"""

from typing import Any

from pydantic import BaseModel

from cognee.infrastructure.llm import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt, render_prompt

GROUNDEDNESS_SYSTEM_PROMPT = "groundedness_eval_system.txt"
GROUNDEDNESS_EVAL_PROMPT = "groundedness_eval_prompt.txt"


class GroundednessEvaluation(BaseModel):
    """Response model containing the groundedness verdict and explanation."""

    grounded: bool
    explanation: str


class GroundednessAdapter:
    def __init__(self):
        """Initialize adapter with the groundedness prompt paths."""
        self.system_prompt_path = GROUNDEDNESS_SYSTEM_PROMPT
        self.eval_prompt_path = GROUNDEDNESS_EVAL_PROMPT

    async def evaluate_groundedness(
        self, question: str, answer: str, context: str
    ) -> dict[str, Any]:
        """Judge whether the answer is supported by the retrieved context.

        Reference-free: there is no golden answer and no correctness score. An answer that
        is unsupported, contradicts the context, or is a non-answer is not grounded.
        """
        args = {"question": question, "answer": answer, "context": context}

        user_prompt = render_prompt(self.eval_prompt_path, args)
        system_prompt = read_query_prompt(self.system_prompt_path)

        if system_prompt is None:
            raise FileNotFoundError(
                f"Groundedness system prompt not found: {self.system_prompt_path}"
            )

        evaluation = await LLMGateway.acreate_structured_output(
            text_input=user_prompt,
            system_prompt=system_prompt,
            response_model=GroundednessEvaluation,
        )

        return {"grounded": evaluation.grounded, "reason": evaluation.explanation}
