"""Blind LLM-judge scoring for final-answer quality.

Two rubrics:

1. ``score_correctness`` — for each of 8 structural properties the ideal
   answer captures, the judge returns a per-property boolean. Final score
   is 0..8. The same rubric applies across all three domains because all
   three reduce to the same structural object.

2. ``score_abstraction`` — binary (Agent 3 only). Does the answer
   explicitly unify multiple apparently-distinct domains under one
   mathematical structure?

Both judgements are blind: the judge sees only the answer text, not the
agent index or the domain.
"""

from pydantic import BaseModel, Field

from cognee.infrastructure.llm.LLMGateway import LLMGateway


PROPERTIES: tuple[tuple[str, str], ...] = (
    (
        "independent_steps",
        "Successive increments / changes / displacements are statistically "
        "independent of each other.",
    ),
    (
        "zero_mean",
        "The increment distribution has mean approximately zero, or a "
        "separate deterministic drift term is explicitly accounted for.",
    ),
    (
        "variance_linear_in_t",
        "The variance (or spread) of the cumulative change grows linearly "
        "with elapsed time / number of steps.",
    ),
    (
        "gaussian_limit",
        "The cumulative change (or its log, for multiplicative processes) "
        "is modelled as normal / Gaussian / bell-shaped.",
    ),
    (
        "markov_property",
        "The future distribution depends only on the current state, not on "
        "the earlier history.",
    ),
    (
        "memoryless",
        "Past observations contribute no predictive information beyond the "
        "current state.",
    ),
    (
        "continuous_or_discrete_time",
        "The answer specifies (or explicitly handles) whether the process "
        "is observed in discrete time steps or as a continuous-time limit.",
    ),
    (
        "drift_term",
        "A deterministic drift / trend component is identified separately "
        "from the stochastic part, or explicitly set to zero.",
    ),
)


class PropertyCheck(BaseModel):
    property_name: str
    captured: bool
    one_line_justification: str


class CorrectnessResult(BaseModel):
    checks: list[PropertyCheck]

    def score(self) -> int:
        return sum(1 for c in self.checks if c.captured)


class AbstractionResult(BaseModel):
    unifies_domains: bool = Field(
        description="True if the answer explicitly identifies that "
        "multiple apparently-distinct observational domains reduce to a "
        "single mathematical structure."
    )
    one_line_justification: str


_CORRECTNESS_SYSTEM = (
    "You are a blind judge evaluating whether a final-answer text captures "
    "specific structural properties of a stochastic process. For EACH "
    "property listed, decide whether the answer captures it. Be strict: a "
    "property counts as captured only if it is explicitly asserted or "
    "clearly implied. Return one PropertyCheck per property, in the given "
    "order, using the property_name verbatim."
)

_ABSTRACTION_SYSTEM = (
    "You are a blind judge. Decide whether the answer text explicitly "
    "identifies that MULTIPLE apparently-distinct observational domains "
    "(different subject matter, different nouns) reduce to the SAME "
    "underlying mathematical structure. Generic statements that 'this "
    "resembles a random process' do NOT count. The answer must make a "
    "cross-domain unification claim to qualify."
)


async def score_correctness(answer: str) -> CorrectnessResult:
    property_list = "\n".join(f"- {name}: {desc}" for name, desc in PROPERTIES)
    prompt = (
        f"ANSWER TEXT TO EVALUATE:\n\n{answer}\n\n"
        f"PROPERTIES TO CHECK (in order):\n{property_list}"
    )
    return await LLMGateway.acreate_structured_output(
        text_input=prompt,
        system_prompt=_CORRECTNESS_SYSTEM,
        response_model=CorrectnessResult,
    )


async def score_abstraction(answer: str) -> AbstractionResult:
    return await LLMGateway.acreate_structured_output(
        text_input=f"ANSWER TEXT TO EVALUATE:\n\n{answer}",
        system_prompt=_ABSTRACTION_SYSTEM,
        response_model=AbstractionResult,
    )
