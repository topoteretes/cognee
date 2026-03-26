import asyncio
import json

from cognee.infrastructure.engine.models.DataPoint import DataPoint
from cognee.infrastructure.llm import LLMGateway
from cognee.shared.data_models import KnowledgeGraph
from cognee.shared.graph_model_utils import graph_model_to_graph_schema


# Define a custom graph model for programming languages.
class FieldType(DataPoint):
    name: str = "Field"


class Field(DataPoint):
    name: str
    is_type: FieldType
    metadata: dict = {"index_fields": ["name"]}


class ProgrammingLanguageType(DataPoint):
    name: str = "Programming Language"


class ProgrammingLanguage(DataPoint):
    name: str
    used_in: list[Field] = []
    is_type: ProgrammingLanguageType
    metadata: dict = {"index_fields": ["name"]}


async def main():
    prompt = """
            You are an expert prompt engineer.

Your task is to generate a production-grade extraction prompt for a knowledge-graph builder, based on an input JSON Schema that defines the target graph model.

## Input
You will receive:
`graph_schema` (JSON Schema) describing node/edge structures, allowed labels/types, required fields, and constraints.

## Output
Return exactly one prompt (plain text, no markdown fences) that can be given to an extraction model.

The generated prompt must:
- Be structured with numbered sections and clear rule headers.
- Enforce strict compliance language.
- Define:
  - What nodes are
  - What edges are
  - Labeling rules
  - ID rules
  - Required properties
  - Date/number normalization
  - Coreference/entity consistency
  - Relationship naming convention
  - Validation/strictness behavior
- Reflect only constraints allowed by `graph_schema`:
  - Allowed node labels/types
  - Allowed relationship types
  - Required fields
  - Property formats/enums/patterns
- Include schema-derived examples only if safely inferable.
- Enforce schema-required fields with zero tolerance:
  - Every field listed in any `required` array in `graph_schema` MUST be present in output objects.
  - Never omit required fields (e.g., `metadata` if required).
  - If a required field value is unknown, use schema-compatible null/placeholder only if the schema allows it; otherwise do not invent invalid structure.
- For missing schema constraints, add sensible defaults:
  - Node IDs must be human-readable strings (never integers unless schema explicitly allows)
  - Every node must include `name` when applicable
  - Use snake_case for relationship names unless schema overrides
  - Dates in `YYYY-MM-DD` when full date available; partial date allowed if schema permits
- Include a “Strict Compliance” section stating that violations are not allowed.
- Avoid contradictions and avoid adding fields not present in schema.

## Generation rules
- Do not explain your reasoning.
- Do not output analysis, JSON, or commentary.
- Output only the final generated extraction prompt text.
- Keep wording precise, imperative, and implementation-ready.
- Include an explicit validation rule in the produced prompt:
  - “Before finalizing output, verify every node/edge includes all schema-required fields; outputs missing required fields are invalid.”

Now generate the extraction prompt from this input:

graph_schema:
{{GRAPH_SCHEMA_JSON}}
"""

    schema_dict = graph_model_to_graph_schema(ProgrammingLanguage)

    graph_model_schema_json = json.dumps(schema_dict)

    print(graph_model_schema_json)

    prompt = prompt.replace("{{GRAPH_SCHEMA_JSON}}", graph_model_schema_json)

    result = await LLMGateway.acreate_structured_output(
        text_input="Generate the extraction prompt now.", system_prompt=prompt, response_model=str
    )

    print(result)


if __name__ == "__main__":
    asyncio.run(main())
