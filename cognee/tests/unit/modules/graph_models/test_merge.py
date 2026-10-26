import pytest
from pydantic import ValidationError

from cognee.modules.graph_models import (
    GraphSchemaMergeError,
    GraphSchemaPatch,
    GraphSchemaSpec,
    merge_graph_schema,
)


def stored_spec() -> dict:
    """A stored document in the frontend's camelCase shape."""
    return {
        "options": {"autoTypeNodes": False},
        "entities": [
            {
                "name": "Person",
                "description": "A person.",
                "fields": [
                    {"kind": "primitive", "name": "age", "primitiveType": "number"},
                    {
                        "kind": "relation",
                        "name": "works_at",
                        "relation": {"targetEntityName": "Organization", "cardinality": "one"},
                    },
                ],
            },
            {"name": "Organization", "fields": []},
        ],
    }


def patch(**body) -> GraphSchemaPatch:
    return GraphSchemaPatch.model_validate(body)


def entity(merged: dict, name: str) -> dict:
    return next(item for item in merged["entities"] if item["name"] == name)


def field_names(merged: dict, name: str) -> list[str]:
    return [item["name"] for item in entity(merged, name)["fields"]]


def test_add_entity_keeps_existing_entities_in_order():
    merged = merge_graph_schema(
        stored_spec(),
        patch(entities=[{"name": "City", "fields": [{"kind": "primitive", "name": "country"}]}]),
    )

    assert [item["name"] for item in merged["entities"]] == ["Person", "Organization", "City"]
    assert field_names(merged, "City") == ["country"]
    assert entity(merged, "Person")["description"] == "A person."


def test_add_field_to_existing_entity_appends_it():
    merged = merge_graph_schema(
        stored_spec(),
        patch(
            entities=[
                {
                    "name": "Person",
                    "fields": [{"kind": "enum", "name": "mood", "enum_values": ["happy"]}],
                }
            ]
        ),
    )

    assert field_names(merged, "Person") == ["age", "works_at", "mood"]
    assert entity(merged, "Person")["description"] == "A person."


def test_replace_field_by_name_swaps_it_whole_in_place():
    merged = merge_graph_schema(
        stored_spec(),
        patch(
            entities=[
                {
                    "name": "Person",
                    "fields": [{"kind": "enum", "name": "age", "enumValues": ["young", "old"]}],
                }
            ]
        ),
    )

    assert field_names(merged, "Person") == ["age", "works_at"]
    age = entity(merged, "Person")["fields"][0]
    assert age["kind"] == "enum"
    assert age["enumValues"] == ["young", "old"]
    assert "primitiveType" not in age


def test_scalars_overwrite_only_when_sent():
    unchanged = merge_graph_schema(stored_spec(), patch(entities=[{"name": "Person"}]))
    assert entity(unchanged, "Person")["description"] == "A person."

    overwritten = merge_graph_schema(
        stored_spec(),
        patch(entities=[{"name": "Person", "description": "Someone.", "indexFields": ["age"]}]),
    )
    assert entity(overwritten, "Person")["description"] == "Someone."
    assert entity(overwritten, "Person")["indexFields"] == ["age"]

    cleared = merge_graph_schema(
        stored_spec(), patch(entities=[{"name": "Person", "description": None}])
    )
    assert "description" not in entity(cleared, "Person")


def test_remove_field():
    merged = merge_graph_schema(
        stored_spec(), patch(remove_fields=[{"entity": "Person", "field": "age"}])
    )

    assert field_names(merged, "Person") == ["works_at"]


def test_remove_field_unknown_names_the_field():
    with pytest.raises(GraphSchemaMergeError, match="'height'.*'Person'"):
        merge_graph_schema(
            stored_spec(), patch(remove_fields=[{"entity": "Person", "field": "height"}])
        )
    with pytest.raises(GraphSchemaMergeError, match="'Robot'"):
        merge_graph_schema(
            stored_spec(), patch(remove_fields=[{"entity": "Robot", "field": "age"}])
        )


def test_remove_entity():
    merged = merge_graph_schema(
        stored_spec(),
        patch(
            remove_fields=[{"entity": "Person", "field": "works_at"}],
            remove_entities=["Organization"],
        ),
    )

    assert [item["name"] for item in merged["entities"]] == ["Person"]


def test_remove_entity_still_targeted_by_relation_names_it():
    with pytest.raises(GraphSchemaMergeError) as excinfo:
        merge_graph_schema(stored_spec(), patch(remove_entities=["Organization"]))

    assert "'Organization'" in str(excinfo.value)
    assert "Person.works_at" in str(excinfo.value)


def test_remove_unknown_entity_rejected():
    with pytest.raises(GraphSchemaMergeError, match="'Robot'"):
        merge_graph_schema(stored_spec(), patch(remove_entities=["Robot"]))


def test_removals_apply_after_upserts():
    """An entity sent in both lists is upserted and then removed (so PATCH ends without it)."""
    merged = merge_graph_schema(
        stored_spec(),
        patch(
            entities=[{"name": "City"}],
            remove_entities=["City"],
        ),
    )

    assert [item["name"] for item in merged["entities"]] == ["Person", "Organization"]


def test_options_shallow_merge():
    kept = merge_graph_schema(stored_spec(), patch(entities=[{"name": "City"}]))
    assert kept["options"] == {"autoTypeNodes": False}

    changed = merge_graph_schema(stored_spec(), patch(options={"autoTypeNodes": True}))
    assert changed["options"] == {"autoTypeNodes": True}


def test_root_overwritten_only_when_sent():
    stored = {**stored_spec(), "root": "Organization"}

    assert merge_graph_schema(stored, patch(entities=[{"name": "City"}]))["root"] == "Organization"
    assert merge_graph_schema(stored, patch(root="Person"))["root"] == "Person"
    assert "root" not in merge_graph_schema(stored, patch(root=None))


def test_patch_on_empty_schema_equals_put_of_the_patch():
    body = {
        "entities": [
            {"name": "Person", "fields": [{"kind": "primitive", "name": "age"}]},
        ],
        "options": {"autoTypeNodes": False},
    }

    from_none = merge_graph_schema(None, patch(**body))
    from_empty = merge_graph_schema({}, patch(**body))
    put_equivalent = GraphSchemaSpec.model_validate(body).model_dump(
        by_alias=True, exclude_none=True
    )

    assert from_none == put_equivalent
    assert from_empty == put_equivalent


def test_patch_on_empty_schema_with_no_entities_rejected():
    """An empty spec has no entities; a patch that adds none fails the spec's min_length."""
    with pytest.raises(ValidationError):
        merge_graph_schema(None, patch(options={"autoTypeNodes": False}))


def test_invalid_merged_spec_rejected_by_spec_validation():
    with pytest.raises(ValidationError, match="plain identifier"):
        merge_graph_schema(stored_spec(), patch(entities=[{"name": "bad-name"}]))

    with pytest.raises(ValidationError, match="undeclared entity 'Nowhere'"):
        merge_graph_schema(
            stored_spec(),
            patch(
                entities=[
                    {
                        "name": "Person",
                        "fields": [
                            {
                                "kind": "relation",
                                "name": "lives_in",
                                "relation": {"targetEntityName": "Nowhere"},
                            }
                        ],
                    }
                ]
            ),
        )


def test_stored_schema_that_is_not_a_spec_cannot_be_patched():
    with pytest.raises(GraphSchemaMergeError, match="replace it with PUT"):
        merge_graph_schema({"nodes": [{"name": "Person"}]}, patch(entities=[{"name": "City"}]))


def test_patch_body_rejects_unknown_keys_and_malformed_fields():
    with pytest.raises(ValidationError):
        patch(entitiez=[])
    with pytest.raises(ValidationError):
        patch(entities=[{"name": "Person", "fields": [{"kind": "enum", "name": "mood"}]}])


def test_merged_document_is_a_valid_spec_in_canonical_camel_case():
    merged = merge_graph_schema(
        stored_spec(),
        patch(
            entities=[
                {
                    "name": "Person",
                    "fields": [{"kind": "primitive", "name": "height", "primitive_type": "number"}],
                }
            ]
        ),
    )

    GraphSchemaSpec.model_validate(merged)
    height = entity(merged, "Person")["fields"][2]
    assert height == {
        "kind": "primitive",
        "name": "height",
        "primitiveType": "number",
        "required": False,
    }
