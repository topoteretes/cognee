import asyncio
import os

import cognee
from cognee.api.v1.visualize.visualize import visualize_graph
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.shared.logging_utils import setup_logging
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import RDFLibOntologyResolver
from cognee.modules.ontology.ontology_config import Config
from os import path
from poc_single_add_datapoints_pipeline import poc_cognify

text_1 = """
1. Audi
Audi is known for its modern designs and advanced technology. Founded in the early 1900s, the brand has earned a reputation for precision engineering and innovation. With features like the Quattro all-wheel-drive system, Audi offers a range of vehicles from stylish sedans to high-performance sports cars.

2. BMW
BMW, short for Bayerische Motoren Werke, is celebrated for its focus on performance and driving pleasure. The company's vehicles are designed to provide a dynamic and engaging driving experience, and their slogan, "The Ultimate Driving Machine," reflects that commitment. BMW produces a variety of cars that combine luxury with sporty performance.

3. Mercedes-Benz
Mercedes-Benz is synonymous with luxury and quality. With a history dating back to the early 20th century, the brand is known for its elegant designs, innovative safety features, and high-quality engineering. Mercedes-Benz manufactures not only luxury sedans but also SUVs, sports cars, and commercial vehicles, catering to a wide range of needs.

4. Porsche
Porsche is a name that stands for high-performance sports cars. Founded in 1931, the brand has become famous for models like the iconic Porsche 911. Porsche cars are celebrated for their speed, precision, and distinctive design, appealing to car enthusiasts who value both performance and style.

5. Volkswagen
Volkswagen, which means "people's car" in German, was established with the idea of making affordable and reliable vehicles accessible to everyone. Over the years, Volkswagen has produced several iconic models, such as the Beetle and the Golf. Today, it remains one of the largest car manufacturers in the world, offering a wide range of vehicles that balance practicality with quality.

Each of these car manufacturer contributes to Germany's reputation as a leader in the global automotive industry, showcasing a blend of innovation, performance, and design excellence.
"""

text_2 = """
1. Apple
Apple is renowned for its innovative consumer electronics and software. Its product lineup includes the iPhone, iPad, Mac computers, and wearables like the Apple Watch. Known for its emphasis on sleek design and user-friendly interfaces, Apple has built a loyal customer base and created a seamless ecosystem that integrates hardware, software, and services.

2. Google
Founded in 1998, Google started as a search engine and quickly became the go-to resource for finding information online. Over the years, the company has diversified its offerings to include digital advertising, cloud computing, mobile operating systems (Android), and various web services like Gmail and Google Maps. Google's innovations have played a major role in shaping the internet landscape.

3. Microsoft
Microsoft Corporation has been a dominant force in software for decades. Its Windows operating system and Microsoft Office suite are staples in both business and personal computing. In recent years, Microsoft has expanded into cloud computing with Azure, gaming with the Xbox platform, and even hardware through products like the Surface line. This evolution has helped the company maintain its relevance in a rapidly changing tech world.

4. Amazon
What began as an online bookstore has grown into one of the largest e-commerce platforms globally. Amazon is known for its vast online marketplace, but its influence extends far beyond retail. With Amazon Web Services (AWS), the company has become a leader in cloud computing, offering robust solutions that power websites, applications, and businesses around the world. Amazon's constant drive for innovation continues to reshape both retail and technology sectors.

5. Meta
Meta, originally known as Facebook, revolutionized social media by connecting billions of people worldwide. Beyond its core social networking service, Meta is investing in the next generation of digital experiences through virtual and augmented reality technologies, with projects like Oculus. The company's efforts signal a commitment to evolving digital interaction and building the metaverse—a shared virtual space where users can connect and collaborate.

Each of these companies has significantly impacted the technology landscape, driving innovation and transforming everyday life through their groundbreaking products and services.
"""


def _edge_key(edge_tuple):
    return (str(edge_tuple[0]), str(edge_tuple[1]), str(edge_tuple[2]))


async def _get_graph_snapshot(label: str):
    graph_engine = await get_graph_engine()
    nodes_data, edges_data = await graph_engine.get_graph_data()

    node_ids = {str(node_id) for node_id, _ in nodes_data}
    edge_keys = {_edge_key(edge) for edge in edges_data}
    node_info_by_id = {str(node_id): node_info for node_id, node_info in nodes_data}
    node_labels = {node_info.get("name") for node_info in node_info_by_id.values() if node_info}

    return {
        "label": label,
        "node_ids": node_ids,
        "edge_keys": edge_keys,
        "node_info_by_id": node_info_by_id,
        "node_labels": node_labels,
        "node_count": len(node_ids),
        "edge_count": len(edge_keys),
    }


def _diff_graph_snapshots(base, other):
    missing_nodes = sorted(set(base["node_labels"]) - set(other["node_labels"]))
    missing_edges = sorted(base["edge_keys"] - other["edge_keys"])

    print("")
    print(f"Graph diff: {base['label']} -> {other['label']}")
    print(f"Nodes: {base['node_count']} -> {other['node_count']}")
    print(f"Edges: {base['edge_count']} -> {other['edge_count']}")
    print(f"Missing nodes in {other['label']}: {len(missing_nodes)}")
    print(f"Missing edges in {other['label']}: {len(missing_edges)}")

    if missing_nodes:
        print("Sample missing node labels (first 20):")
        for node_label in missing_nodes[:20]:
            print(f"  {node_label}")

    if missing_edges:
        print("Sample missing edges (first 20):")
        for source, target, relation in missing_edges[:20]:
            print(f"  {source} -[{relation}]-> {target}")


async def main(use_poc):
    # Step 1: Reset data and system state
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    # Step 2: Add text
    text_list = [text_1, text_2]
    await cognee.add(text_list)

    # Step 3: Create knowledge graph

    ontology_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ontology_input_example/basic_ontology.owl"
    )

    # Create full config structure manually
    config: Config = {
        "ontology_config": {
            "ontology_resolver": RDFLibOntologyResolver(ontology_file=ontology_path)
        }
    }

    if use_poc:
        await poc_cognify(config=config, use_single_add_datapoints_poc=True)
    else:
        await cognee.cognify(config=config)

    graph_visualization_path = path.join(
        path.dirname(__file__),
        f"results/{'poc_' if use_poc else ''}cognify_result_text.html",
    )

    await visualize_graph(graph_visualization_path)
    return await _get_graph_snapshot("poc" if use_poc else "default")


if __name__ == "__main__":
    logger = setup_logging()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        default_snapshot = loop.run_until_complete(main(use_poc=False))
        poc_snapshot = loop.run_until_complete(main(use_poc=True))
        _diff_graph_snapshots(default_snapshot, poc_snapshot)
        print("POC")
        _diff_graph_snapshots(poc_snapshot, default_snapshot)
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
