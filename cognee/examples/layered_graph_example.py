"""
Layered Knowledge Graph Example for Cognee.

This example demonstrates how to build layered knowledge graphs and evaluate
them using the Cognee framework.
"""

import asyncio
import logging
import sys
from typing import List, Dict, Any

from cognee.shared.data_models import KnowledgeGraph, Node, Edge
from cognee.modules.graph.layered_graph_builder import LayeredGraphBuilder, convert_to_layered_graph
from cognee.modules.graph.layered_graph_service import LayeredGraphService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Try to import evaluation components, but continue if not available
try:
    from cognee.eval_framework.evaluation.layered_graph_eval_adapter import LayeredGraphEvalAdapter
    EVAL_AVAILABLE = True
    logger.info("Evaluation framework available. Full example will run.")
except ImportError:
    EVAL_AVAILABLE = False
    logger.warning("Evaluation framework not available. Running with limited functionality.")


def create_car_brands_graph() -> KnowledgeGraph:
    """
    Create a simple knowledge graph about car brands.
    
    Returns:
        Knowledge graph with car brands
    """
    nodes = [
        Node(
            id="brand_audi",
            name="Audi",
            type="CarBrand",
            description="Audi is a German luxury car manufacturer."
        ),
        Node(
            id="brand_bmw",
            name="BMW",
            type="CarBrand",
            description="BMW is a German luxury car manufacturer."
        ),
        Node(
            id="country_germany",
            name="Germany",
            type="Country",
            description="Germany is a country in Central Europe."
        ),
        Node(
            id="city_munich",
            name="Munich",
            type="City",
            description="Munich is a city in Germany."
        ),
        Node(
            id="city_ingolstadt",
            name="Ingolstadt",
            type="City",
            description="Ingolstadt is a city in Germany."
        )
    ]
    
    edges = [
        Edge(
            source_node_id="brand_audi",
            target_node_id="country_germany",
            relationship_name="HEADQUARTERED_IN"
        ),
        Edge(
            source_node_id="brand_bmw",
            target_node_id="country_germany",
            relationship_name="HEADQUARTERED_IN"
        ),
        Edge(
            source_node_id="brand_audi",
            target_node_id="city_ingolstadt",
            relationship_name="BASED_IN"
        ),
        Edge(
            source_node_id="brand_bmw",
            target_node_id="city_munich",
            relationship_name="BASED_IN"
        ),
        Edge(
            source_node_id="city_munich",
            target_node_id="country_germany",
            relationship_name="LOCATED_IN"
        ),
        Edge(
            source_node_id="city_ingolstadt",
            target_node_id="country_germany",
            relationship_name="LOCATED_IN"
        )
    ]
    
    return KnowledgeGraph(
        nodes=nodes,
        edges=edges,
        name="Car Brands Graph",
        description="A knowledge graph about car brands and their locations."
    )


def create_car_models_graph() -> KnowledgeGraph:
    """
    Create a knowledge graph about car models.
    
    Returns:
        Knowledge graph with car models
    """
    nodes = [
        Node(
            id="model_a4",
            name="A4",
            type="CarModel",
            description="The Audi A4 is a line of compact executive cars produced since 1994."
        ),
        Node(
            id="model_a6",
            name="A6",
            type="CarModel",
            description="The Audi A6 is an executive car made by Audi."
        ),
        Node(
            id="model_3series",
            name="3 Series",
            type="CarModel",
            description="The BMW 3 Series is a line of compact executive cars."
        ),
        Node(
            id="model_5series",
            name="5 Series",
            type="CarModel",
            description="The BMW 5 Series is an executive car manufactured by BMW."
        ),
        Node(
            id="brand_audi",
            name="Audi",
            type="CarBrand",
            description="Audi is a German luxury car manufacturer."
        ),
        Node(
            id="brand_bmw",
            name="BMW",
            type="CarBrand",
            description="BMW is a German luxury car manufacturer."
        )
    ]
    
    edges = [
        Edge(
            source_node_id="model_a4",
            target_node_id="brand_audi",
            relationship_name="MADE_BY"
        ),
        Edge(
            source_node_id="model_a6",
            target_node_id="brand_audi",
            relationship_name="MADE_BY"
        ),
        Edge(
            source_node_id="model_3series",
            target_node_id="brand_bmw",
            relationship_name="MADE_BY"
        ),
        Edge(
            source_node_id="model_5series",
            target_node_id="brand_bmw",
            relationship_name="MADE_BY"
        )
    ]
    
    return KnowledgeGraph(
        nodes=nodes,
        edges=edges,
        name="Car Models Graph",
        description="A knowledge graph about car models and their brands."
    )


def create_car_specs_graph() -> KnowledgeGraph:
    """
    Create a knowledge graph about car specifications.
    
    Returns:
        Knowledge graph with car specifications
    """
    nodes = [
        Node(
            id="model_a4",
            name="A4",
            type="CarModel",
            description="The Audi A4 is a line of compact executive cars produced since 1994."
        ),
        Node(
            id="model_a6",
            name="A6",
            type="CarModel",
            description="The Audi A6 is an executive car made by Audi."
        ),
        Node(
            id="model_3series",
            name="3 Series",
            type="CarModel",
            description="The BMW 3 Series is a line of compact executive cars."
        ),
        Node(
            id="model_5series",
            name="5 Series",
            type="CarModel",
            description="The BMW 5 Series is an executive car manufactured by BMW."
        ),
        Node(
            id="engine_2_0tdi",
            name="2.0 TDI",
            type="Engine",
            description="2.0-liter turbocharged diesel engine."
        ),
        Node(
            id="engine_3_0tfsi",
            name="3.0 TFSI",
            type="Engine",
            description="3.0-liter turbocharged gasoline engine."
        ),
        Node(
            id="engine_2_0i",
            name="2.0i",
            type="Engine",
            description="2.0-liter gasoline engine."
        ),
        Node(
            id="engine_3_0i",
            name="3.0i",
            type="Engine",
            description="3.0-liter gasoline engine."
        )
    ]
    
    edges = [
        Edge(
            source_node_id="model_a4",
            target_node_id="engine_2_0tdi",
            relationship_name="HAS_ENGINE_OPTION"
        ),
        Edge(
            source_node_id="model_a6",
            target_node_id="engine_2_0tdi",
            relationship_name="HAS_ENGINE_OPTION"
        ),
        Edge(
            source_node_id="model_a6",
            target_node_id="engine_3_0tfsi",
            relationship_name="HAS_ENGINE_OPTION"
        ),
        Edge(
            source_node_id="model_3series",
            target_node_id="engine_2_0i",
            relationship_name="HAS_ENGINE_OPTION"
        ),
        Edge(
            source_node_id="model_5series",
            target_node_id="engine_2_0i",
            relationship_name="HAS_ENGINE_OPTION"
        ),
        Edge(
            source_node_id="model_5series",
            target_node_id="engine_3_0i",
            relationship_name="HAS_ENGINE_OPTION"
        )
    ]
    
    return KnowledgeGraph(
        nodes=nodes,
        edges=edges,
        name="Car Specifications Graph",
        description="A knowledge graph about car specifications and engine options."
    )


async def build_layered_car_graph() -> Dict[str, Any]:
    """
    Build a layered car knowledge graph.
    
    Returns:
        Dictionary with the layered graph and layer IDs
    """
    # Create the builder
    builder = LayeredGraphBuilder(
        name="Layered Car Knowledge Graph",
        description="A layered knowledge graph about cars, with brands, models, and specifications."
    )
    
    # Create the base layer with car brands
    base_layer_id = builder.create_layer(
        name="Car Brands Layer",
        description="Base layer with car brands and geographical information",
        layer_type="base"
    )
    
    # Add car brands subgraph to the base layer
    brands_graph = create_car_brands_graph()
    builder.add_subgraph_to_layer(base_layer_id, brands_graph)
    
    # Create the models layer
    models_layer_id = builder.create_layer(
        name="Car Models Layer",
        description="Layer with car models information",
        layer_type="enrichment",
        parent_layers=[base_layer_id]
    )
    
    # Add car models subgraph to the models layer
    models_graph = create_car_models_graph()
    builder.add_subgraph_to_layer(models_layer_id, models_graph)
    
    # Create the specifications layer
    specs_layer_id = builder.create_layer(
        name="Car Specifications Layer",
        description="Layer with car specifications and engine information",
        layer_type="enrichment",
        parent_layers=[models_layer_id]
    )
    
    # Add car specifications subgraph to the specifications layer
    specs_graph = create_car_specs_graph()
    builder.add_subgraph_to_layer(specs_layer_id, specs_graph)
    
    # Build the layered graph
    layered_graph = builder.build()
    
    # Return the graph and layer IDs
    return {
        "layered_graph": layered_graph,
        "base_layer_id": base_layer_id,
        "models_layer_id": models_layer_id,
        "specs_layer_id": specs_layer_id
    }


async def analyze_layered_graph(layered_graph_data: Dict[str, Any]) -> None:
    """
    Analyze a layered graph using the LayeredGraphService.
    
    Args:
        layered_graph_data: Dictionary with layered graph and layer IDs
    """
    layered_graph = layered_graph_data["layered_graph"]
    
    # Analyze layer dependencies
    dependencies = await LayeredGraphService.analyze_layer_dependencies(layered_graph)
    logger.info("Layer Dependencies:")
    logger.info(f"Root layers: {dependencies['root_layers']}")
    logger.info(f"Leaf layers: {dependencies['leaf_layers']}")
    logger.info(f"Max depth: {dependencies['max_depth']}")
    
    # Calculate metrics for each layer
    metrics = await LayeredGraphService.calculate_layer_metrics(layered_graph)
    logger.info("\nLayer Metrics:")
    for layer_id, layer_metrics in metrics.items():
        logger.info(f"Layer {layer_id}:")
        logger.info(f"  Node count: {layer_metrics['node_count']}")
        logger.info(f"  Edge count: {layer_metrics['edge_count']}")
        logger.info(f"  Cumulative node count: {layer_metrics['cumulative_node_count']}")
        logger.info(f"  Cumulative edge count: {layer_metrics['cumulative_edge_count']}")
        logger.info(f"  Node contribution ratio: {layer_metrics['node_contribution_ratio']:.2f}")
        logger.info(f"  Edge contribution ratio: {layer_metrics['edge_contribution_ratio']:.2f}")
    
    # Compare base layer and models layer
    base_layer_id = layered_graph_data["base_layer_id"]
    models_layer_id = layered_graph_data["models_layer_id"]
    diff_result = await LayeredGraphService.diff_layers(
        layered_graph, base_layer_id, models_layer_id
    )
    logger.info("\nDifference between Base Layer and Models Layer:")
    logger.info(f"Added nodes: {len(diff_result['added_nodes'])}")
    logger.info(f"Added edges: {len(diff_result['added_edges'])}")
    
    # Extract a specific relationship type
    filtered_graph = await LayeredGraphService.filter_graph_by_relationship_types(
        layered_graph, ["MADE_BY"], include_only=True
    )
    logger.info("\nFiltered Graph (MADE_BY relationships only):")
    logger.info(f"Node count: {len(filtered_graph.nodes)}")
    logger.info(f"Edge count: {len(filtered_graph.edges)}")


async def evaluate_layered_graph(layered_graph_data: Dict[str, Any]) -> None:
    """
    Evaluate a layered knowledge graph using the LayeredGraphEvalAdapter.
    
    Args:
        layered_graph_data: Dictionary with layered graph and layer IDs
    """
    if not EVAL_AVAILABLE:
        logger.warning("Skipping evaluation as evaluation framework is not available.")
        return
        
    layered_graph = layered_graph_data["layered_graph"]
    
    # Create evaluation questions
    questions = [
        "What car brands are headquartered in Germany?",
        "Which city is Audi based in?",
        "What models does Audi make?",
        "What engine options are available for the Audi A6?",
        "Where is the BMW 5 Series manufactured?"
    ]
    
    # Expected answers (simplified for the example)
    expected_answers = [
        "Audi and BMW are car brands headquartered in Germany.",
        "Audi is based in Ingolstadt, Germany.",
        "Audi makes the A4 and A6 models.",
        "The Audi A6 has 2.0 TDI and 3.0 TFSI engine options.",
        "The BMW 5 Series is manufactured by BMW in Munich, Germany."
    ]
    
    # Create the evaluation adapter
    evaluator = LayeredGraphEvalAdapter(evaluator="direct_llm")
    
    # Define layer IDs to evaluate
    layer_ids = [
        layered_graph_data["base_layer_id"],
        layered_graph_data["models_layer_id"],
        layered_graph_data["specs_layer_id"]
    ]
    
    # Evaluate the layered graph
    logger.info("\nEvaluating layered graph...")
    eval_results = await evaluator.evaluate_layered_graph(
        layered_graph=layered_graph,
        questions=questions,
        expected_answers=expected_answers,
        eval_metrics=["correctness", "relevance"],
        layer_ids=layer_ids
    )
    
    # Print evaluation results
    logger.info("\nEvaluation Results:")
    
    # Per layer results
    logger.info("\nPer Layer Results:")
    for layer_id, layer_results in eval_results["per_layer"].items():
        if layer_results.get("is_empty", False):
            logger.info(f"Layer {layer_id}: Empty layer")
            continue
            
        logger.info(f"Layer {layer_id}:")
        for metric, score in layer_results["metrics"].items():
            logger.info(f"  {metric}: {score:.4f}")
    
    # Cumulative results
    logger.info("\nCumulative Results:")
    for layer_id, layer_results in eval_results["cumulative"].items():
        if layer_results.get("is_empty", False):
            logger.info(f"Cumulative {layer_id}: Empty layer")
            continue
            
        logger.info(f"Cumulative {layer_id}:")
        for metric, score in layer_results["metrics"].items():
            logger.info(f"  {metric}: {score:.4f}")
    
    # Layer improvements
    logger.info("\nLayer Improvements:")
    for layer_id, improvements in eval_results["layer_improvements"].items():
        logger.info(f"Layer {layer_id} improvements:")
        for metric, improvement in improvements.items():
            # Format as percentage with +/- sign
            sign = "+" if improvement >= 0 else ""
            logger.info(f"  {metric}: {sign}{improvement:.2%}")


async def demonstrate_converting_regular_graph() -> None:
    """
    Demonstrate converting a regular KnowledgeGraph to a LayeredKnowledgeGraph.
    """
    # Create a regular knowledge graph
    regular_graph = create_car_brands_graph()
    
    # Convert to a layered graph
    layered_graph = await convert_to_layered_graph(
        knowledge_graph=regular_graph,
        layer_name="Brands Base Layer",
        layer_description="Converted from regular graph",
        graph_name="Converted Brands Graph",
        graph_description="Demonstration of converting a regular graph to a layered graph"
    )
    
    # Print information about the converted graph
    logger.info("\nConverted Regular Graph to Layered Graph:")
    logger.info(f"Graph name: {layered_graph.name}")
    logger.info(f"Number of layers: {len(layered_graph.layers)}")
    if layered_graph.layers:
        layer = layered_graph.layers[0]
        logger.info(f"Layer name: {layer.name}")
        logger.info(f"Layer description: {layer.description}")
        
    # Get the graph from the single layer
    layer_graph = layered_graph.get_layer_graph(layered_graph.layers[0].id)
    logger.info(f"Layer node count: {len(layer_graph.nodes)}")
    logger.info(f"Layer edge count: {len(layer_graph.edges)}")


async def main() -> None:
    """
    Main function to run the layered graph example.
    """
    logger.info("===== Layered Knowledge Graph Example =====")
    
    logger.info("\nBuilding layered car knowledge graph...")
    layered_graph_data = await build_layered_car_graph()
    
    logger.info("\nAnalyzing layered graph...")
    await analyze_layered_graph(layered_graph_data)
    
    logger.info("\nDemonstrating conversion of regular graph to layered graph...")
    await demonstrate_converting_regular_graph()
    
    if EVAL_AVAILABLE:
        logger.info("\nEvaluating layered graph...")
        await evaluate_layered_graph(layered_graph_data)
    else:
        logger.info("\nSkipping evaluation (framework not available).")
        # Display graph structure details instead
        layered_graph = layered_graph_data["layered_graph"]
        logger.info(f"\nLayered Graph Structure:")
        for i, layer in enumerate(layered_graph.layers):
            logger.info(f"Layer {i+1}: {layer.name} (ID: {layer.id})")
            layer_graph = layered_graph.get_layer_graph(layer.id)
            logger.info(f"  Nodes: {len(layer_graph.nodes)}")
            logger.info(f"  Edges: {len(layer_graph.edges)}")
            
            # Show node and relationship types
            node_types = {}
            for node in layer_graph.nodes:
                if node.type not in node_types:
                    node_types[node.type] = 0
                node_types[node.type] += 1
                
            rel_types = {}
            for edge in layer_graph.edges:
                if edge.relationship_name not in rel_types:
                    rel_types[edge.relationship_name] = 0
                rel_types[edge.relationship_name] += 1
                
            logger.info(f"  Node types: {node_types}")
            logger.info(f"  Relationship types: {rel_types}")
            
            # If not the first layer, show parent layers
            if layer.parent_layers:
                logger.info(f"  Parent layers: {layer.parent_layers}")
    
    logger.info("\nExample completed!")


if __name__ == "__main__":
    asyncio.run(main()) 