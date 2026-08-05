"""Reference-free judge for real user questions: coverage and groundedness.

Real questions have no golden answer, so they cannot be scored for correctness. This
adapter returns the two things that ARE knowable without one, as independent booleans:

* ``answered`` — did the memory supply the information asked for, or decline? This is the
  COVERAGE signal. A synthetic question cannot produce it, because synthetic questions are
  generated from chunks that exist by construction; a question the tenant actually asked is
  the only input that can reveal the memory holding nothing relevant.
* ``grounded`` — is what the answer does assert supported by the retrieved context? This is
  the HALLUCINATION signal.

They are separate because they are independent, and because overloading one boolean with
both did not work: the prompt previously asked ``grounded`` to be false for a refusal, and
the judge overrode it, since a refusal's claim ("the context does not contain this") is
genuinely traceable to the context. A coverage gap is answered=false/grounded=true; a
hallucination is answered=true/grounded=false.
"""

from typing import Any

from pydantic import BaseModel, Field

from cognee.infrastructure.llm import LLMGateway
from cognee.infrastructure.llm.prompts import read_query_prompt, render_prompt

GROUNDEDNESS_SYSTEM_PROMPT = "groundedness_eval_system.txt"
GROUNDEDNESS_EVAL_PROMPT = "groundedness_eval_prompt.txt"


class GroundednessEvaluation(BaseModel):
    """Response model: the coverage and groundedness verdicts, plus one explanation."""

    answered: bool = Field(
        description=(
            "True when the answer supplies the information asked for. False when it "
            "declines or reports it cannot answer — the coverage signal."
        )
    )
    grounded: bool = Field(
        description=(
            "True when every substantive claim the answer makes is supported by the "
            "retrieved context — the hallucination signal."
        )
    )
    explanation: str


class GroundednessAdapter:
    def __init__(self):
        """Initialize adapter with the groundedness prompt paths."""
        self.system_prompt_path = GROUNDEDNESS_SYSTEM_PROMPT
        self.eval_prompt_path = GROUNDEDNESS_EVAL_PROMPT

    async def evaluate_groundedness(
        self, question: str, answer: str, context: str
    ) -> dict[str, Any]:
        """Judge coverage and groundedness in ONE call.

        Reference-free: there is no golden answer and no correctness score. Both verdicts
        come from the same call rather than two, so adding the coverage signal costs no
        extra tokens beyond the one field.
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

        return {
            "answered": evaluation.answered,
            "grounded": evaluation.grounded,
            "reason": evaluation.explanation,
        }
