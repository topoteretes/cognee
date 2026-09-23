from cognee.infrastructure.engine import DataPoint


class OntologyProperty(DataPoint):
    """An ontology property (``owl:ObjectProperty`` / ``owl:DatatypeProperty``) as a node.

    Written by schema alignment when a column realizes it (``cust_id`` realizes
    ``hasCustomerId``); linked to its domain and range classes with ``domain`` /
    ``range`` edges. Same deterministic-id convention as ``EntityType`` so every
    ingestion path meets on one node per property.
    """

    name: str
    description: str
    property_kind: str = "datatype"  # "object" | "datatype"
    metadata: dict = {"index_fields": ["name"], "identity_fields": ["name"]}
