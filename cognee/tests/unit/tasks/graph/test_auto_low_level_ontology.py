import pytest
from typing import get_args

from cognee.modules.graph.utils import get_graph_from_model
from cognee.tasks.graph.auto_low_level_ontology import (
    GeneratedLowLevelDataPointClass,
    GeneratedLowLevelDataPointModel,
    GeneratedLowLevelField,
    GeneratedLowLevelRelation,
    build_low_level_extraction_model,
    instantiate_low_level_datapoints,
)


@pytest.mark.asyncio
async def test_generated_low_level_model_instantiates_datapoints_and_edges():
    generated_model = GeneratedLowLevelDataPointModel(
        classes=[
            GeneratedLowLevelDataPointClass(
                class_name="Company",
                scalar_fields=[
                    GeneratedLowLevelField(name="name"),
                    GeneratedLowLevelField(name="text"),
                ],
                index_field="text",
                identity_field="name",
            ),
            GeneratedLowLevelDataPointClass(
                class_name="Person",
                scalar_fields=[
                    GeneratedLowLevelField(name="name"),
                    GeneratedLowLevelField(name="text"),
                    GeneratedLowLevelField(name="title"),
                ],
                relation_fields=[
                    GeneratedLowLevelRelation(name="works_at", target_class="Company")
                ],
                index_field="text",
                identity_field="name",
            ),
        ]
    )
    extraction_model, _ = build_low_level_extraction_model(generated_model)
    extraction = extraction_model(
        company_items=[{"local_id": "company_acme", "name": "Acme", "text": "Acme"}],
        person_items=[
            {
                "local_id": "person_ada",
                "name": "Ada",
                "text": "Ada, Engineer",
            }
        ],
        relationships=[
            {
                "source_class": "Person",
                "source_id": "person_ada",
                "relationship_name": "works_at",
                "target_class": "Company",
                "target_id": "company_acme",
            }
        ],
    )

    datapoints = instantiate_low_level_datapoints(extraction, generated_model)
    person = next(datapoint for datapoint in datapoints if datapoint.type == "Person")

    nodes, edges = await get_graph_from_model(person)

    assert person.metadata == {"index_fields": ["text"], "identity_fields": ["name", "text"]}
    assert not hasattr(person, "title")
    assert {node.type for node in nodes} == {"Person", "Company", "EntityType"}
    assert sorted((edge[2], edge[3]["relationship_name"]) for edge in edges) == [
        ("is_a", "is_a"),
        ("is_a", "is_a"),
        ("works_at", "works_at"),
    ]
    assert any(
        node.type == "EntityType" and node.name == "Person" and node.description == "Person"
        for node in nodes
    )


def test_generated_low_level_model_removes_resume_sections_as_scalar_fields():
    generated_model = GeneratedLowLevelDataPointModel(
        classes=[
            GeneratedLowLevelDataPointClass(
                class_name="Person",
                scalar_fields=[
                    GeneratedLowLevelField(name="name"),
                    GeneratedLowLevelField(name="text"),
                    GeneratedLowLevelField(name="email"),
                    GeneratedLowLevelField(name="education"),
                    GeneratedLowLevelField(name="experience"),
                    GeneratedLowLevelField(name="skills"),
                    GeneratedLowLevelField(name="summary"),
                ],
                index_field="text",
                identity_field="email",
            )
        ]
    )

    extraction_model, _ = build_low_level_extraction_model(generated_model)
    person_record = get_args(extraction_model.model_fields["person_items"].annotation)[0]

    assert set(person_record.model_fields) == {"local_id", "name", "text"}


def test_generated_low_level_model_adds_name_to_text_only_nodes():
    generated_model = GeneratedLowLevelDataPointModel(
        classes=[
            GeneratedLowLevelDataPointClass(
                class_name="Skill",
                scalar_fields=[GeneratedLowLevelField(name="text")],
                index_field="text",
                identity_field="text",
            )
        ]
    )

    extraction_model, _ = build_low_level_extraction_model(generated_model)
    skill_record = get_args(extraction_model.model_fields["skill_items"].annotation)[0]
    extraction = extraction_model(
        skill_items=[{"local_id": "skill_azure", "name": "Azure", "text": "Azure"}],
        relationships=[],
    )
    datapoints = instantiate_low_level_datapoints(extraction, generated_model)

    assert set(skill_record.model_fields) == {"local_id", "name", "text"}
    assert datapoints[0].name == "Azure"
    assert datapoints[0].text == "Azure"
    assert datapoints[0].metadata == {"index_fields": ["text"], "identity_fields": ["name", "text"]}
    assert datapoints[0].is_a[1][0].name == "Skill"
    assert datapoints[0].is_a[1][0].type == "EntityType"
    assert datapoints[0].is_a[1][0].description == "Skill"


def test_generated_low_level_model_forbids_summary_class():
    generated_model = GeneratedLowLevelDataPointModel(
        classes=[
            GeneratedLowLevelDataPointClass(
                class_name="Document",
                scalar_fields=[
                    GeneratedLowLevelField(name="name"),
                    GeneratedLowLevelField(name="text"),
                ],
                relation_fields=[
                    GeneratedLowLevelRelation(name="has_summary", target_class="Summary")
                ],
            ),
            GeneratedLowLevelDataPointClass(
                class_name="Summary",
                scalar_fields=[
                    GeneratedLowLevelField(name="name"),
                    GeneratedLowLevelField(name="text"),
                ],
            ),
        ]
    )

    extraction_model, _ = build_low_level_extraction_model(generated_model)

    assert "summary_items" not in extraction_model.model_fields
    document_record = get_args(extraction_model.model_fields["document_items"].annotation)[0]
    assert set(document_record.model_fields) == {"local_id", "name", "text"}
