#!/usr/bin/env python3
"""
Example script demonstrating the layered knowledge graph pipeline.

This script shows how to use the layered knowledge graph pipeline to process a dataset and
extract layered knowledge graphs from it. The example demonstrates:

1. Creating a dataset with sample text documents
2. Running a layered graph pipeline with customized layers and pipeline steps
3. Creating and storing a layered knowledge graph using the cognify API

A layered knowledge graph is a graph structure where nodes and relationships are organized
into distinct layers, each with a specific purpose or perspective. Layers can build upon
each other, allowing for rich, multi-dimensional knowledge representation.

Layers can include:
- Base layers: containing fundamental entities and relationships
- Enrichment layers: adding additional context, classifications, or attributes
- Inference layers: containing derived knowledge or insights

This example demonstrates how to define custom layers, run the pipeline, and store the results.

Key concepts:
- Layer: A distinct perspective or aspect of the knowledge graph
- Layer Configuration: Defines how each layer should be extracted and what it represents
- Pipeline Configuration: Defines the processing steps for the layered graph
- Graph Database: The backend storage for the layered graph (Neo4j, NetworkX, etc.)

The example works with any configured graph database adapter in your system.
"""

import asyncio
import logging
import sys
import uuid
from pathlib import Path

import cognee
from cognee.api.v1.cognify import cognify_layered_graph
from cognee.modules.data.models import Dataset, Data
from cognee.modules.users.methods import get_default_user
from cognee.infrastructure.databases.graph import get_graph_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

async def create_example_dataset():
    """
    Create an example dataset with some text documents.
    
    This function demonstrates how to create a Dataset with multiple Data objects,
    each containing sample text about different technology companies.
    
    Returns:
        tuple: A tuple containing (dataset, data_documents)
            - dataset: The Dataset object
            - data_documents: List of Data objects in the dataset
    """
    # Get default user
    user = await get_default_user()
    
    # Create dataset
    dataset = Dataset(
        id=uuid.uuid4(),
        name="Layered Graph Example",
        owner_id=user.id
    )
    
    # Create some example documents
    sample_texts = [
        """
        Apple Inc. is an American multinational technology company headquartered in Cupertino, California. 
        Tim Cook is the CEO of Apple. The company develops and sells consumer electronics, computer software, and online services. 
        Apple's hardware products include the iPhone, iPad, Mac, Apple Watch, and Apple TV. Its software includes macOS, iOS, iPadOS, 
        watchOS, and tvOS. Apple also offers online services such as iCloud, Apple Music, and Apple TV+.
        """,
        
        """
        Microsoft Corporation is an American multinational technology company headquartered in Redmond, Washington. 
        Satya Nadella is the CEO of Microsoft. The company develops, manufactures, licenses, supports, and sells computer software, 
        consumer electronics, personal computers, and related services. Its best-known software products are the Microsoft Windows 
        line of operating systems, the Microsoft Office suite, and the Edge web browser.
        """,
        
        """
        Google LLC is an American multinational technology company that specializes in Internet-related services and products. 
        Sundar Pichai is the CEO of Google. The company's products include online advertising technologies, search engine, cloud 
        computing, software, and hardware. Google was founded by Larry Page and Sergey Brin while they were PhD students at 
        Stanford University.
        """
    ]
    
    data_documents = []
    for i, text in enumerate(sample_texts):
        # Create a Data object for each text
        data = Data(
            id=uuid.uuid4(),
            name=f"Sample Text {i+1}",
            extension=".txt",
            mime_type="text/plain",
            raw_data_location=f"/tmp/sample_text_{i+1}.txt",
            owner_id=user.id,
            external_metadata={"text_content": text}
        )
        data_documents.append(data)
    
    return dataset, data_documents

async def run_layered_graph_pipeline(dataset):
    """
    Run the layered knowledge graph pipeline on a dataset.
    
    This function demonstrates how to configure and run a layered knowledge graph pipeline
    with custom layer and pipeline configurations.
    
    Args:
        dataset: The dataset to process
        
    Returns:
        dict: The pipeline execution results
    """
    # Custom layer configuration
    # Each layer represents a different perspective or aspect of the knowledge graph
    layer_config = [
        {
            # Base layer for entities - the foundation of the graph
            "name": "Entities Layer",
            "description": "Basic entities extracted from the content",
            "layer_type": "base",
            "prompt": "Extract people, organizations, products, and locations from the content."
        },
        {
            # Relationships layer - connects entities from the base layer
            "name": "Relationships Layer",
            "description": "Relationships between entities",
            "layer_type": "relationships",
            "prompt": "Extract relationships between the entities, such as employment, ownership, and development relationships."
        },
        {
            # Categories layer - provides classification and grouping
            "name": "Categories Layer",
            "description": "Categorization of entities",
            "layer_type": "categories",
            "prompt": "Categorize the entities into types like 'Person', 'Organization', 'Product', and 'Location'."
        }
    ]
    
    # Custom pipeline configuration
    # Defines the processing steps for the layered graph
    pipeline_config = [
        {
            # Analysis step - extracts basic information
            "type": "analyze",
            "description": "Analyze the layered graph"
        },
        {
            # Enrichment step - adds additional context and information
            "type": "enrich",
            "description": "Add industry classifications",
            "enrichment_type": "classification",
            "content": "Classify the technology companies by their main industry sectors (e.g., consumer electronics, software, internet services)."
        },
        {
            # Storage step - persists the graph in the database
            "type": "store",
            "description": "Store the graph in the database"
        }
    ]
    
    # Run the pipeline
    # This will process the dataset, extract the layered graph, and store it in the database
    logger.info(f"Running layered graph pipeline with {len(layer_config)} layers...")
    result = await cognify_layered_graph(
        datasets=[dataset],
        layer_config=layer_config,
        pipeline_config=pipeline_config
    )
    
    return result

async def main():
    """Main function that orchestrates the example."""
    logger.info("=== Layered Knowledge Graph Pipeline Example ===")
    
    try:
        # Display which graph database is being used
        # The example works with any configured graph database (Neo4j, NetworkX, etc.)
        graph_config = get_graph_config()
        logger.info(f"Using graph database provider: {graph_config.graph_database_provider}")
        
        # Step 1: Create a dataset with sample documents
        logger.info("Creating example dataset...")
        dataset, data_documents = await create_example_dataset()
        logger.info(f"Created dataset with {len(data_documents)} documents")
        
        # Step 2: Run the layered graph pipeline on the dataset
        logger.info("\nRunning layered graph pipeline...")
        result = await run_layered_graph_pipeline(dataset)
        logger.info("Pipeline execution completed successfully!")
        
        # Step 3: Display the results of the pipeline execution
        logger.info("\nLayered Graph Pipeline Results:")
        if result:
            logger.info(f"Created layered graphs for dataset: {dataset.name}")
            if isinstance(result, list):
                logger.info(f"Tasks executed: {len(result)}")
                for task_result in result:
                    if hasattr(task_result, 'name'):
                        logger.info(f"  - Task: {task_result.name}")
            elif isinstance(result, dict):
                for key, value in result.items():
                    logger.info(f"  - {key}: {value}")
        
        # The layered graph is now stored in the configured graph database
        # and can be accessed using the LayeredGraphDBAdapter or through the API
        
        logger.info("\n=== Example completed successfully! ===")
        
        # Provide instructions about pruning
        logger.info("\nTo clean up the database after running this example, you can use:")
        logger.info("await cognee.prune.prune_data()")
        logger.info("await cognee.prune.prune_system()")
        
    except Exception as e:
        logger.error(f"Error in example: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main()) 