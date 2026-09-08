"""
Define a custom graph model in plain JSON and let cognee turn it into a
Pydantic model — no model classes to write.

The JSON shape (a "graph schema spec") declares entities, their fields, and
relations with cardinality; it is the same document the cognee UI graph-model
editor produces. `graph_model_from_spec` validates it, compiles it to JSON
Schema, and generates a DataPoint-derived Pydantic class you can pass as
`graph_model=` to `cognify()` or `remember()`.

Notes:
- `identity_fields` (Python-side extension, default `["name"]`) makes nodes
  with the same identity values merge into one graph node across chunks and
  runs. Set `"identity_fields": []` on an entity to opt out.
- Custom graph models skip ontology grounding and the extra dedup passes of
  the default KnowledgeGraph path, and do not compose with
  `functional_relationships`.

Requires a configured LLM (e.g. LLM_API_KEY) for the cognify step.
"""

import asyncio

import cognee
from cognee.low_level import graph_model_from_spec

PEOPLE_SPEC = {
    "root": "Person",
    "entities": [
        {
            "name": "Person",
            "description": "A person mentioned in the text.",
            "fields": [
                {
                    "kind": "primitive",
                    "name": "role",
                    "primitive_type": "string",
                    "description": "What the person does.",
                },
                {
                    "kind": "relation",
                    "name": "works_at",
                    "relation": {"target_entity_name": "Organization", "cardinality": "one"},
                },
                {
                    "kind": "relation",
                    "name": "collaborates_with",
                    "relation": {"target_entity_name": "Person", "cardinality": "many"},
                },
            ],
        },
        {
            "name": "Organization",
            "description": "A company, lab, or institution.",
            "fields": [
                {"kind": "primitive", "name": "field_of_work", "primitive_type": "string"},
            ],
        },
    ],
}

TEXT = """
Ada Lovelace worked at the Analytical Engine project alongside Charles Babbage.
Grace Hopper worked at Remington Rand, where she collaborated with the UNIVAC team.
"""


async def main():
    await cognee.forget(everything=True)

    # JSON in, Pydantic model out.
    PeopleGraph = graph_model_from_spec(PEOPLE_SPEC)

    await cognee.add(TEXT, dataset_name="people_from_json")
    await cognee.cognify(datasets=["people_from_json"], graph_model=PeopleGraph)

    results = await cognee.search(
        query_text="Who worked where, and with whom?",
        datasets=["people_from_json"],
    )
    for result in results:
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
