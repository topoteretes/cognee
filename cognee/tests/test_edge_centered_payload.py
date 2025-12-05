"""
End-to-end integration test for edge-centered payload and triplet embeddings.

"""

import os
import pathlib
import cognee
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.modules.search.types import SearchType
from cognee.shared.logging_utils import get_logger
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver
from cognee.modules.ontology.ontology_config import Config

logger = get_logger()

text_data = """
Apple is a technology company that produces the iPhone, iPad, and Mac computers.
The company is known for its innovative products and ecosystem integration.

Microsoft develops the Windows operating system and Office productivity suite.
They are also major players in cloud computing with Azure.

Google created the Android operating system and provides search engine services.
The company is a leader in artificial intelligence and machine learning.
"""

ontology_content = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xmlns="http://example.org/tech#"
         xml:base="http://example.org/tech">

    <owl:Ontology rdf:about="http://example.org/tech"/>

    <!-- Classes -->
    <owl:Class rdf:ID="Company"/>
    <owl:Class rdf:ID="TechnologyCompany"/>
    <owl:Class rdf:ID="Product"/>
    <owl:Class rdf:ID="Software"/>
    <owl:Class rdf:ID="Hardware"/>
    <owl:Class rdf:ID="Service"/>

    <rdf:Description rdf:about="#TechnologyCompany">
        <rdfs:subClassOf rdf:resource="#Company"/>
        <rdfs:comment>A company operating in the technology sector.</rdfs:comment>
    </rdf:Description>

    <rdf:Description rdf:about="#Software">
        <rdfs:subClassOf rdf:resource="#Product"/>
        <rdfs:comment>Software products and applications.</rdfs:comment>
    </rdf:Description>

    <rdf:Description rdf:about="#Hardware">
        <rdfs:subClassOf rdf:resource="#Product"/>
        <rdfs:comment>Physical hardware products.</rdfs:comment>
    </rdf:Description>

    <!-- Individuals -->
    <TechnologyCompany rdf:ID="apple">
        <rdfs:label>Apple</rdfs:label>
    </TechnologyCompany>

    <TechnologyCompany rdf:ID="microsoft">
        <rdfs:label>Microsoft</rdfs:label>
    </TechnologyCompany>

    <TechnologyCompany rdf:ID="google">
        <rdfs:label>Google</rdfs:label>
    </TechnologyCompany>

    <Hardware rdf:ID="iphone">
        <rdfs:label>iPhone</rdfs:label>
    </Hardware>

    <Software rdf:ID="windows">
        <rdfs:label>Windows</rdfs:label>
    </Software>

    <Software rdf:ID="android">
        <rdfs:label>Android</rdfs:label>
    </Software>

</rdf:RDF>"""


async def main():
    data_directory_path = str(
        pathlib.Path(
            os.path.join(
                pathlib.Path(__file__).parent,
                ".data_storage/test_edge_centered_payload",
            )
        ).resolve()
    )
    cognee_directory_path = str(
        pathlib.Path(
            os.path.join(
                pathlib.Path(__file__).parent,
                ".cognee_system/test_edge_centered_payload",
            )
        ).resolve()
    )

    cognee.config.data_root_directory(data_directory_path)
    cognee.config.system_root_directory(cognee_directory_path)

    dataset_name = "tech_companies"

    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    await cognee.add(data=text_data, dataset_name=dataset_name)

    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".owl", delete=False) as f:
        f.write(ontology_content)
        ontology_file_path = f.name

    try:
        logger.info(f"Loading ontology from: {ontology_file_path}")
        config: Config = {
            "ontology_config": {
                "ontology_resolver": RDFLibOntologyResolver(ontology_file=ontology_file_path)
            }
        }

        await cognee.cognify(datasets=[dataset_name], config=config)
        graph_engine = await get_graph_engine()
        nodes_phase2, edges_phase2 = await graph_engine.get_graph_data()

        vector_engine = get_vector_engine()
        triplets_phase2 = await vector_engine.search(
            query_text="technology", limit=None, collection_name="Triplet_text"
        )

        assert len(triplets_phase2) == len(edges_phase2), (
            f"Triplet embeddings and number of edges do not match. Vector db contains {len(triplets_phase2)} edge triplets while graph db contains {len(edges_phase2)} edges."
        )

        search_results_phase2 = await cognee.search(
            query_type=SearchType.TRIPLET_COMPLETION,
            query_text="What products does Apple make?",
        )

        assert search_results_phase2 is not None, (
            "Search should return results for triplet embeddings in simple ontology use case."
        )

    finally:
        if os.path.exists(ontology_file_path):
            os.unlink(ontology_file_path)


if __name__ == "__main__":
    import asyncio
    from cognee.shared.logging_utils import setup_logging

    setup_logging()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
