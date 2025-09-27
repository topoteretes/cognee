from cognee.modules.ontology.rdf_xml.OntologyResolver import OntologyResolver
from cognee.modules.engine.models.Entity import Entity
from cognee.modules.engine.models.EntityType import EntityType
from cognee.tasks.storage.add_data_points import add_data_points
from cognee.infrastructure.databases.vector.get_vector_engine import get_vector_engine

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
        texts_to_embed = []
        # Ingest classes as EntityType
        for key, uri in self.resolver.lookup.get("classes", {}).items():
            datapoints.append(
                EntityType(
                    name=key,
                    description=str(uri),
                    metadata={"category": "class", "uri": str(uri)}
                )
            )
            texts_to_embed.append(f"{key} {str(uri)}")
        # Ingest individuals as Entity
        for key, uri in self.resolver.lookup.get("individuals", {}).items():
            datapoints.append(
                Entity(
                    name=key,
                    description=str(uri),
                    metadata={"category": "individual", "uri": str(uri)}
                )
            )
            texts_to_embed.append(f"{key} {str(uri)}")
        # Batch embedding
        if texts_to_embed:
            embedding_engine = get_vector_engine().embedding_engine
            embeddings = await embedding_engine.embed_text(texts_to_embed)
            for dp, emb in zip(datapoints, embeddings):
                dp.embedding = emb
        await add_data_points(datapoints, update_edge_collection=True)
        self.nodes = [dp.dict() for dp in datapoints]
        self.edges = []  # Edges handled by add_data_points
    # No return needed; ingestion is complete
