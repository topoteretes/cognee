from cognee.modules.ontology.rdf_xml.OntologyResolver import OntologyResolver
from cognee.modules.engine.models.Entity import Entity
from cognee.modules.engine.models.EntityType import EntityType
from cognee.tasks.storage.add_data_points import add_data_points

class OntologyPipeline:
    """
    Dedicated pipeline for OWL/RDF ontology ingestion and processing.
    """
    def __init__(self, ontology_file: str):
        self.ontology_file = ontology_file
        self.resolver = OntologyResolver(ontology_file=ontology_file)
        self.nodes = []
        self.edges = []

    async def ingest(self):
        datapoints = []
        # Ingest classes as EntityType
        for key, uri in self.resolver.lookup.get("classes", {}).items():
            datapoints.append(
                EntityType(
                    name=key,
                    description=str(uri),
                    metadata={"category": "class", "uri": str(uri)}
                )
            )
        # Ingest individuals as Entity
        for key, uri in self.resolver.lookup.get("individuals", {}).items():
            datapoints.append(
                Entity(
                    name=key,
                    description=str(uri),
                    metadata={"category": "individual", "uri": str(uri)}
                )
            )
        await add_data_points(datapoints, update_edge_collection=True)
        self.nodes = [dp.dict() for dp in datapoints]
        self.edges = []  # Edges handled by add_data_points
        return {"nodes": self.nodes, "edges": self.edges}
