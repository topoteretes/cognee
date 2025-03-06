from cognee.infrastructure.engine import DataPoint


class Ontology(DataPoint):
    name: str
    ontology_origin_type: str

    metadata: dict = {"index_fields": ["name"]}
