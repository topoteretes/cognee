from types import SimpleNamespace
from typing import get_args

import pytest

from cognee.modules.graph.utils import get_graph_from_model
from cognee.tasks.graph.auto_low_level_ontology import (
    AutoLowLevelCanonicalOntology,
    GeneratedCanonicalTypeRelation,
    GeneratedCanonicalType,
    GeneratedLowLevelDataPointClass,
    GeneratedLowLevelDataPointModel,
    GeneratedLowLevelField,
    GeneratedLowLevelRelation,
    GeneratedLowLevelCanonicalOntology,
    GeneratedSubclassRelation,
    LLMGateway,
    _canonical_ontology_json,
    _normalize_canonical_ontology,
    build_low_level_extraction_model,
    generate_low_level_model_from_chunks,
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


def test_canonical_ontology_normalization_rejects_duplicates_self_links_and_cycles():
    ontology = GeneratedLowLevelCanonicalOntology(
        types=[
            GeneratedCanonicalType(name="car model", aliases=["vehicle model"]),
            GeneratedCanonicalType(name="CarModel", aliases=["automobile model"]),
        ],
        subclass_of=[
            GeneratedSubclassRelation(child_type="car model", parent_type="product"),
            GeneratedSubclassRelation(child_type="CarModel", parent_type="Product"),
            GeneratedSubclassRelation(child_type="Product", parent_type="Product"),
            GeneratedSubclassRelation(child_type="Product", parent_type="CarModel"),
        ],
    )

    normalized = _normalize_canonical_ontology(ontology, dataset_id="dataset-1")

    assert [type_spec.name for type_spec in normalized.types] == ["CarModel", "Product"]
    assert normalized.types[0].aliases == ["VehicleModel", "AutomobileModel"]
    assert normalized.subclass_of == [
        GeneratedSubclassRelation(child_type="CarModel", parent_type="Product")
    ]
    assert all(type_spec.dataset_id == "dataset-1" for type_spec in normalized.types)


def test_canonical_ontology_normalization_rejects_generic_sink_parents():
    ontology = GeneratedLowLevelCanonicalOntology(
        types=[
            GeneratedCanonicalType(name="AutoEntity"),
            GeneratedCanonicalType(name="Candidate"),
            GeneratedCanonicalType(name="Document"),
            GeneratedCanonicalType(name="Person"),
        ],
        subclass_of=[
            GeneratedSubclassRelation(child_type="Candidate", parent_type="Document"),
            GeneratedSubclassRelation(child_type="Candidate", parent_type="AutoEntity"),
            GeneratedSubclassRelation(child_type="AutoEntity", parent_type="Entity"),
            GeneratedSubclassRelation(child_type="Candidate", parent_type="Person"),
        ],
    )

    normalized = _normalize_canonical_ontology(ontology)

    assert "AutoEntity" not in [type_spec.name for type_spec in normalized.types]
    assert normalized.subclass_of == [
        GeneratedSubclassRelation(child_type="Candidate", parent_type="Person")
    ]


def test_canonical_ontology_normalization_keeps_real_type_relations_only():
    ontology = GeneratedLowLevelCanonicalOntology(
        types=[
            GeneratedCanonicalType(name="Person"),
            GeneratedCanonicalType(name="Organization"),
            GeneratedCanonicalType(name="ListModel"),
            GeneratedCanonicalType(name="Project"),
        ],
        subclass_of=[
            GeneratedSubclassRelation(child_type="ListModel", parent_type="Project")
        ],
        type_relations=[
            GeneratedCanonicalTypeRelation(
                source_type="Person",
                relationship_type="works at",
                target_type="Organization",
            ),
            GeneratedCanonicalTypeRelation(
                source_type="ListModel",
                relationship_type="describes",
                target_type="Project",
            ),
            GeneratedCanonicalTypeRelation(
                source_type="Person",
                relationship_type="related_to",
                target_type="Project",
            ),
        ],
    )

    normalized = _normalize_canonical_ontology(ontology)

    assert "ListModel" not in [type_spec.name for type_spec in normalized.types]
    assert normalized.subclass_of == []
    assert normalized.type_relations == [
        GeneratedCanonicalTypeRelation(
            source_type="Person",
            relationship_type="works_at",
            target_type="Organization",
        )
    ]


@pytest.mark.asyncio
async def test_canonical_ontology_creates_subclass_edges_between_entity_types():
    generated_model = GeneratedLowLevelDataPointModel(
        classes=[
            GeneratedLowLevelDataPointClass(
                class_name="CarModel",
                scalar_fields=[
                    GeneratedLowLevelField(name="name"),
                    GeneratedLowLevelField(name="text"),
                ],
            )
        ]
    )
    ontology = GeneratedLowLevelCanonicalOntology(
        types=[
            GeneratedCanonicalType(name="CarModel"),
            GeneratedCanonicalType(name="Product"),
        ],
        subclass_of=[
            GeneratedSubclassRelation(child_type="CarModel", parent_type="Product")
        ],
    )
    extraction_model, _ = build_low_level_extraction_model(generated_model)
    extraction = extraction_model(
        car_model_items=[
            {
                "local_id": "model_1",
                "name": "Roadster LX",
                "text": "Roadster LX electric compact car",
            }
        ],
        relationships=[],
    )

    datapoints = instantiate_low_level_datapoints(extraction, generated_model, ontology=ontology)
    car_model = datapoints[0]
    nodes, edges = await get_graph_from_model(car_model)

    assert car_model.type == "CarModel"
    assert datapoints[1].type == "EntityType"
    assert datapoints[1].name == "CarModel"
    assert car_model.is_a[1][0].name == "CarModel"
    assert ("subclass_of", "subclass_of") in [
        (edge[2], edge[3]["relationship_name"]) for edge in edges
    ]
    assert any(node.type == "EntityType" and node.name == "Product" for node in nodes)


@pytest.mark.asyncio
async def test_canonical_ontology_creates_semantic_type_relation_edges():
    generated_model = GeneratedLowLevelDataPointModel(
        classes=[
            GeneratedLowLevelDataPointClass(
                class_name="Person",
                scalar_fields=[
                    GeneratedLowLevelField(name="name"),
                    GeneratedLowLevelField(name="text"),
                ],
            )
        ]
    )
    ontology = GeneratedLowLevelCanonicalOntology(
        types=[
            GeneratedCanonicalType(name="Person"),
            GeneratedCanonicalType(name="Organization"),
        ],
        type_relations=[
            GeneratedCanonicalTypeRelation(
                source_type="Person",
                relationship_type="works_at",
                target_type="Organization",
            )
        ],
    )
    extraction_model, _ = build_low_level_extraction_model(generated_model)
    extraction = extraction_model(
        person_items=[
            {
                "local_id": "person_1",
                "name": "Ada",
                "text": "Ada works at Acme.",
            }
        ],
        relationships=[],
    )

    datapoints = instantiate_low_level_datapoints(extraction, generated_model, ontology=ontology)
    type_relation_root = next(
        datapoint
        for datapoint in datapoints
        if datapoint.type == "EntityType" and datapoint.name == "Person"
    )
    nodes, edges = await get_graph_from_model(type_relation_root)

    assert ("works_at", "works_at") in [
        (edge[2], edge[3]["relationship_name"]) for edge in edges
    ]
    assert any(node.type == "EntityType" and node.name == "Organization" for node in nodes)


@pytest.mark.asyncio
async def test_low_level_generation_receives_canonical_ontology_context(monkeypatch):
    captured = {}
    generated_model = GeneratedLowLevelDataPointModel(
        classes=[
            GeneratedLowLevelDataPointClass(
                class_name="Candidate",
                scalar_fields=[
                    GeneratedLowLevelField(name="name"),
                    GeneratedLowLevelField(name="text"),
                ],
            )
        ]
    )

    async def fake_structured_output(**kwargs):
        captured["system_prompt"] = kwargs["system_prompt"]
        return generated_model

    monkeypatch.setattr(LLMGateway, "acreate_structured_output", fake_structured_output)

    await generate_low_level_model_from_chunks(
        [SimpleNamespace(text="Ada is a senior engineer.")],
        canonical_ontologies=[
            GeneratedLowLevelCanonicalOntology(
                types=[
                    GeneratedCanonicalType(name="Candidate"),
                    GeneratedCanonicalType(name="Person"),
                ],
                subclass_of=[
                    GeneratedSubclassRelation(child_type="Candidate", parent_type="Person")
                ],
                type_relations=[
                    GeneratedCanonicalTypeRelation(
                        source_type="Candidate",
                        relationship_type="applies_to",
                        target_type="Role",
                    )
                ],
            )
        ],
    )

    assert "Dataset canonical type ontology context" in captured["system_prompt"]
    assert "Candidate subclass_of Person" in captured["system_prompt"]
    assert "Candidate applies_to Role" in captured["system_prompt"]


@pytest.mark.asyncio
async def test_canonical_vector_retrieval_filters_by_dataset(monkeypatch):
    class FakeVectorResult:
        def __init__(self, payload):
            self.payload = payload

    class FakeVectorEngine:
        async def search(self, collection_name, query_text, limit, include_payload):
            assert collection_name == "low_level_canonical_text"
            assert query_text == "car description"
            assert include_payload is True
            assert limit == 20
            matching_model = GeneratedLowLevelDataPointModel(
                classes=[
                    GeneratedLowLevelDataPointClass(
                        class_name="CarModel",
                        scalar_fields=[
                            GeneratedLowLevelField(name="name"),
                            GeneratedLowLevelField(name="text"),
                        ],
                    )
                ]
            )
            other_model = GeneratedLowLevelDataPointModel(
                classes=[
                    GeneratedLowLevelDataPointClass(
                        class_name="Candidate",
                        scalar_fields=[
                            GeneratedLowLevelField(name="name"),
                            GeneratedLowLevelField(name="text"),
                        ],
                    )
                ]
            )
            return [
                FakeVectorResult(
                    {
                        "dataset_id": "other-dataset",
                        "structure_json": other_model.model_dump_json(),
                    }
                ),
                FakeVectorResult(
                    {
                        "dataset_id": "dataset-1",
                        "structure_json": matching_model.model_dump_json(),
                        "ontology_json": _canonical_ontology_json(
                            GeneratedLowLevelCanonicalOntology(
                                types=[GeneratedCanonicalType(name="CarModel")]
                            )
                        ),
                    }
                ),
            ]

    monkeypatch.setattr(
        "cognee.tasks.graph.auto_low_level_ontology.get_vector_engine",
        lambda: FakeVectorEngine(),
    )

    mode = AutoLowLevelCanonicalOntology(top_k=2)
    models, ontologies = await mode._get_nearby_models_from_vector_store(
        SimpleNamespace(text="car description"),
        "dataset-1",
    )

    assert [model.classes[0].class_name for model in models] == ["CarModel"]
    assert [ontology.types[0].name for ontology in ontologies] == ["CarModel"]
