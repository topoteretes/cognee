from typing import Optional, List, Literal, Union
import pandas as pd
from cognee.modules.visualization.cognee_network_visualization import (
    cognee_network_visualization,
)
from cognee.modules.visualization.embedding_atlas_export import (
    export_embeddings_to_atlas,
    get_embeddings_for_atlas,
)
from cognee.infrastructure.databases.graph import get_graph_engine
from cognee.shared.logging_utils import get_logger, setup_logging, ERROR


import asyncio


logger = get_logger()


async def visualize_graph(
    destination_file_path: str = None,
    mode: Literal["network", "atlas", "atlas_component"] = "network",
    collections: Optional[List[str]] = None,
    limit: Optional[int] = None,
    compute_projection: bool = True
) -> Union[str, pd.DataFrame]:
    """
    Visualize Cognee's knowledge graph using different visualization modes.
    
    Parameters:
    -----------
    destination_file_path (str, optional): Path where to save the output file.
                                         For network mode: saves HTML file
                                         For atlas mode: saves parquet file
                                         Not used for atlas_component mode
    mode (str): Visualization mode:
               - "network": Interactive HTML graph visualization
               - "atlas": Export to parquet for embedding-atlas CLI
               - "atlas_component": Return DataFrame for Streamlit component
    collections (List[str], optional): For atlas modes - list of collections to export
    limit (int, optional): For atlas modes - maximum number of embeddings to export per collection
    compute_projection (bool): For atlas_component mode - whether to compute 2D projection
    
    Returns:
    --------
    Union[str, pd.DataFrame]: 
        - str: Path to generated file (network and atlas modes)
        - pd.DataFrame: DataFrame for Streamlit component (atlas_component mode)
    
    Usage:
    ------
    # Traditional network visualization
    await visualize_graph()
    
    # Embedding atlas CLI export
    await visualize_graph(mode="atlas", destination_file_path="my_embeddings.parquet")
    
    # Streamlit component DataFrame
    df = await visualize_graph(mode="atlas_component")
    
    # Then use in Streamlit:
    from embedding_atlas.streamlit import embedding_atlas
    selection = embedding_atlas(df, text="text", x="projection_x", y="projection_y")
    """
    
    if mode == "atlas":
        # Export embeddings for atlas CLI visualization
        output_path = destination_file_path or "cognee_embeddings.parquet"
        result_path = await export_embeddings_to_atlas(
            output_path=output_path,
            collections=collections,
            limit=limit
        )
        
        logger.info(f"Embeddings exported to: {result_path}")
        logger.info(f"To visualize with Embedding Atlas, run: embedding-atlas {result_path}")
        
        return result_path
    
    elif mode == "atlas_component":
        # Return DataFrame for Streamlit component
        df = await get_embeddings_for_atlas(
            collections=collections,
            limit=limit,
            compute_projection=compute_projection
        )
        
        logger.info(f"Prepared DataFrame with {len(df)} embeddings for Streamlit component")
        if compute_projection and 'projection_x' in df.columns:
            logger.info("DataFrame includes 2D projection coordinates")
        
        return df
    
    else:
        # Traditional network visualization
        graph_engine = await get_graph_engine()
        graph_data = await graph_engine.get_graph_data()

        graph = await cognee_network_visualization(graph_data, destination_file_path)

        if destination_file_path:
            logger.info(f"The HTML file has been stored at path: {destination_file_path}")
        else:
            logger.info(
                "The HTML file has been stored on your home directory! Navigate there with cd ~"
            )

        return graph


if __name__ == "__main__":
    logger = setup_logging(log_level=ERROR)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(visualize_graph())
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
