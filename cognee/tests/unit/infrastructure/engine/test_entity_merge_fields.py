"""Tests for the entity-canonicalization audit fields on the Entity model
(issue #3629): merged_aliases and merge_confidence. They must default to None,
serialize into node attributes when set, and never affect the UUID5 identity or
the embedding index_fields."""

from fastapi.encoders import jsonable_encoder

from cognee.modules.engine.models import Entity


class TestEntityMergeFields:
    def test_defaults_are_none(self):
        entity = Entity(name="Alice", description="an engineer")
        assert entity.merged_aliases is None
        assert entity.merge_confidence is None

    def test_setting_fields_does_not_change_identity(self):
        # The id is derived from the identity_fields (name) only; stamping the
        # merge audit fields must not alter it.
        baseline_id = Entity(name="Alice", description="an engineer").id

        entity = Entity(name="Alice", description="an engineer")
        entity.merged_aliases = ["Alicia", "Alyce"]
        entity.merge_confidence = 0.93

        assert entity.id == baseline_id
        assert entity.id == Entity.id_for("Alice")

    def test_fields_excluded_from_index_and_identity(self):
        entity = Entity(name="Alice", description="an engineer")
        assert entity.metadata["index_fields"] == ["name"]
        assert entity.metadata["identity_fields"] == ["name"]
        assert "merged_aliases" not in entity.metadata["index_fields"]
        assert "merge_confidence" not in entity.metadata["index_fields"]
        assert "merged_aliases" not in entity.metadata["identity_fields"]
        assert "merge_confidence" not in entity.metadata["identity_fields"]

    def test_fields_serialize_into_node_attributes(self):
        # add_data_points persists nodes via jsonable_encoder(node); the audit
        # fields must ride along in that serialization when set.
        entity = Entity(name="Alice", description="an engineer")
        entity.merged_aliases = ["Alicia"]
        entity.merge_confidence = 0.87

        encoded = jsonable_encoder(entity)
        assert encoded["merged_aliases"] == ["Alicia"]
        assert encoded["merge_confidence"] == 0.87

    def test_normal_setattr_path_now_used(self):
        # Regression guard for commit 3's _safe_setattr: since these are now real
        # declared fields, plain setattr must succeed (no ValueError to catch).
        entity = Entity(name="Alice", description="an engineer")
        entity.merged_aliases = ["Alicia"]  # would raise before the fields existed
        entity.merge_confidence = 0.5
        assert entity.merged_aliases == ["Alicia"]
        assert entity.merge_confidence == 0.5
