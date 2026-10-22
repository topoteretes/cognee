"""
The default relationship model presort checks for, written in the JSON
graph-model DSL (``cognee.modules.graph_models``).

The relation-field names on the root entity drive which relationship sections
``build_report`` emits: a custom spec passed as ``relationship_spec=`` that
omits e.g. ``contains_pii`` disables PII edge reporting. This is the same spec
shape ``graph_model_from_spec`` compiles into a Pydantic graph model, so the
report's relationships can also be written into the graph itself.
"""

DEFAULT_PRESORT_SPEC = {
    "root": "FileRecord",
    "entities": [
        {
            "name": "FileRecord",
            "description": "A file discovered during presort scanning.",
            "index_fields": ["name"],
            "identity_fields": ["content_hash"],
            "fields": [
                {"kind": "primitive", "name": "path", "primitive_type": "string", "required": True},
                {"kind": "primitive", "name": "extension", "primitive_type": "string"},
                {"kind": "primitive", "name": "mime_type", "primitive_type": "string"},
                {"kind": "primitive", "name": "content_hash", "primitive_type": "string"},
                {"kind": "primitive", "name": "size_bytes", "primitive_type": "number"},
                {
                    "kind": "relation",
                    "name": "duplicate_of",
                    "relation": {"target_entity_name": "FileRecord", "cardinality": "many"},
                },
                {
                    "kind": "relation",
                    "name": "version_of",
                    "relation": {"target_entity_name": "FileRecord", "cardinality": "many"},
                },
                {
                    "kind": "relation",
                    "name": "belongs_to_group",
                    "relation": {"target_entity_name": "FileGroup", "cardinality": "one"},
                },
                {
                    "kind": "relation",
                    "name": "contains_pii",
                    "relation": {"target_entity_name": "PiiTag", "cardinality": "many"},
                },
            ],
        },
        {
            "name": "FileGroup",
            "description": "A proposed dataset grouping of related files.",
            "identity_fields": ["name"],
            "fields": [
                {"kind": "primitive", "name": "reason", "primitive_type": "string"},
                {"kind": "primitive", "name": "proposed_dataset", "primitive_type": "string"},
            ],
        },
        {
            "name": "PiiTag",
            "description": "A category of potential personal data detected in a file.",
            "identity_fields": ["name"],
            "fields": [
                {"kind": "enum", "name": "severity", "enum_values": ["low", "medium", "high"]},
            ],
        },
    ],
}
