import os
import pandas as pd
from typing import List, Optional, Dict, Any, Union, Tuple
from pathlib import Path

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.databases.vector import get_vector_engine
from cognee.infrastructure.databases.vector.models.ScoredResult import ScoredResult

logger = get_logger("EmbeddingAtlasExport")


async def get_embeddings_for_atlas(
    collections: Optional[List[str]] = None,
    limit: Optional[int] = None,
    compute_projection: bool = True
) -> pd.DataFrame:
    """
    Get embeddings from Cognee vector database as a DataFrame for use with 
    Embedding Atlas Streamlit component.
    
    Parameters:
    -----------
    collections (List[str], optional): List of collection names to export.
                                      If None, exports from all collections
    limit (int, optional): Maximum number of embeddings to export per collection.
                          If None, exports all embeddings
    compute_projection (bool): Whether to compute 2D projection for visualization.
                              If False, returns raw embeddings only.
    
    Returns:
    --------
    pd.DataFrame: DataFrame ready for use with embedding_atlas() Streamlit component
                 Contains columns: id, text, collection, embedding, dim_0, dim_1, ...
                 If compute_projection=True, also includes: projection_x, projection_y
    
    Usage with Streamlit:
    --------------------
    ```python
    import streamlit as st
    from embedding_atlas.streamlit import embedding_atlas
    from embedding_atlas.projection import compute_text_projection
    import cognee
    
    # Get embeddings DataFrame
    df = await cognee.get_embeddings_for_atlas()
    
    # Use with Embedding Atlas Streamlit component
    selection = embedding_atlas(
        df, 
        text="text",
        x="projection_x", 
        y="projection_y",
        show_table=True
    )
    ```
    """
    
    vector_engine = get_vector_engine()
    
    # Get all collections if none specified
    if collections is None:
        collections = await _get_all_collections(vector_engine)
        logger.info(f"Found {len(collections)} collections: {collections}")
    
    all_data = []
    
    for collection_name in collections:
        logger.info(f"Getting embeddings from collection: {collection_name}")
        
        try:
            # Get all data points from the collection with embeddings
            collection_data = await _get_collection_embeddings(
                vector_engine, collection_name, limit
            )
            
            if collection_data:
                all_data.extend(collection_data)
                logger.info(f"Retrieved {len(collection_data)} embeddings from {collection_name}")
            else:
                logger.warning(f"No data found in collection: {collection_name}")
                
        except Exception as e:
            logger.error(f"Error getting embeddings from collection {collection_name}: {e}")
            continue
    
    if not all_data:
        logger.warning("No embeddings found")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Compute 2D projection if requested
    if compute_projection and 'embedding' in df.columns:
        try:
            from embedding_atlas.projection import compute_text_projection
            
            # Compute projection using the embedding_atlas library
            df = compute_text_projection(
                df, 
                text="text",
                x="projection_x", 
                y="projection_y",
                neighbors="neighbors"
            )
            logger.info("Computed 2D projection for embeddings")
            
        except ImportError:
            logger.warning("embedding-atlas not installed. Install with: pip install embedding-atlas")
            logger.info("Returning DataFrame without projection")
        except Exception as e:
            logger.error(f"Error computing projection: {e}")
            logger.info("Returning DataFrame without projection")
    
    logger.info(f"Prepared DataFrame with {len(df)} embeddings for Atlas component")
    return df


async def export_embeddings_to_atlas(
    output_path: str = None,
    collections: Optional[List[str]] = None,
    limit: Optional[int] = None
) -> str:
    """
    Export embeddings and metadata from Cognee vector database to parquet format
    compatible with Embedding Atlas.
    
    Parameters:
    -----------
    output_path (str, optional): Path where to save the parquet file. 
                                If None, saves to current directory as 'cognee_embeddings.parquet'
    collections (List[str], optional): List of collection names to export.
                                      If None, exports from all collections
    limit (int, optional): Maximum number of embeddings to export per collection.
                          If None, exports all embeddings
    
    Returns:
    --------
    str: Path to the generated parquet file
    
    Usage:
    ------
    After calling this function, you can use the generated parquet file with embedding-atlas:
    ```
    embedding-atlas your-dataset.parquet
    ```
    """
    
    if output_path is None:
        output_path = "cognee_embeddings.parquet"
    
    vector_engine = get_vector_engine()
    
    # Get all collections if none specified
    if collections is None:
        collections = await _get_all_collections(vector_engine)
        logger.info(f"Found {len(collections)} collections: {collections}")
    
    all_data = []
    
    for collection_name in collections:
        logger.info(f"Exporting embeddings from collection: {collection_name}")
        
        try:
            # Get all data points from the collection with embeddings
            collection_data = await _get_collection_embeddings(
                vector_engine, collection_name, limit
            )
            
            if collection_data:
                all_data.extend(collection_data)
                logger.info(f"Exported {len(collection_data)} embeddings from {collection_name}")
            else:
                logger.warning(f"No data found in collection: {collection_name}")
                
        except Exception as e:
            logger.error(f"Error exporting from collection {collection_name}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No embeddings found to export")
    
    # Convert to DataFrame and save as parquet
    df = pd.DataFrame(all_data)
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_parquet(output_path, index=False)
    
    logger.info(f"Successfully exported {len(all_data)} embeddings to {output_path}")
    logger.info(f"You can now visualize with: embedding-atlas {output_path}")
    
    return str(output_path)


async def _get_all_collections(vector_engine) -> List[str]:
    """Get all collection names from the vector database."""
    try:
        # LanceDB specific method
        if hasattr(vector_engine, 'get_connection'):
            connection = await vector_engine.get_connection()
            if hasattr(connection, 'table_names'):
                return await connection.table_names()
        
        # ChromaDB specific method
        if hasattr(vector_engine, 'get_collection_names'):
            return await vector_engine.get_collection_names()
        elif hasattr(vector_engine, 'list_collections'):
            collections = await vector_engine.list_collections()
            return [col.name if hasattr(col, 'name') else str(col) for col in collections]
        else:
            logger.warning("Vector engine doesn't support listing collections")
            return []
    except Exception as e:
        logger.error(f"Error getting collections: {e}")
        return []


async def _get_collection_embeddings(
    vector_engine, 
    collection_name: str, 
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Get all embeddings and metadata from a specific collection."""
    
    try:
        # First check if collection exists
        if not await vector_engine.has_collection(collection_name):
            logger.warning(f"Collection {collection_name} does not exist")
            return []
        
        collection_data = []
        
        # Get collection object to work with directly
        collection = await vector_engine.get_collection(collection_name)
        
        if collection is None:
            logger.warning(f"Could not get collection object for {collection_name}")
            return []
        
        # Strategy 1: LanceDB specific - query all data with vectors
        if hasattr(collection, 'query') and hasattr(collection, 'to_pandas'):
            try:
                logger.info(f"Using LanceDB query method for {collection_name}")
                
                # Query all data from LanceDB table
                query = collection.query()
                if limit:
                    query = query.limit(limit)
                
                results_df = await query.to_pandas()
                
                if not results_df.empty:
                    for _, row in results_df.iterrows():
                        item = {
                            'id': str(row.get('id', '')),
                            'collection': collection_name
                        }
                        
                        # Extract text from payload
                        payload = row.get('payload', {})
                        if isinstance(payload, dict):
                            item['text'] = _extract_text_from_payload(payload)
                            
                            # Add payload metadata
                            for key, value in payload.items():
                                if key not in ['id', 'text', 'embedding', 'vector']:
                                    item[f'meta_{key}'] = value
                        else:
                            item['text'] = str(payload) if payload else ''
                        
                        # Add embedding vector if available
                        if 'vector' in row and row['vector'] is not None:
                            embedding = row['vector']
                            if hasattr(embedding, 'tolist'):
                                embedding = embedding.tolist()
                            elif not isinstance(embedding, list):
                                embedding = list(embedding)
                            
                            item['embedding'] = embedding
                            # Add individual embedding dimensions as columns for atlas
                            for j, val in enumerate(embedding):
                                item[f'dim_{j}'] = float(val)
                        
                        collection_data.append(item)
                    
                    logger.info(f"Exported {len(collection_data)} embeddings from LanceDB table {collection_name}")
                    return collection_data
                    
            except Exception as e:
                logger.debug(f"LanceDB query failed for {collection_name}: {e}")
        
        # Strategy 2: ChromaDB specific - collection.get()
        if hasattr(collection, 'get'):
            try:
                logger.info(f"Using ChromaDB get method for {collection_name}")
                results = await collection.get(
                    include=["metadatas", "embeddings", "documents"]
                )
                
                if results and 'ids' in results:
                    for i, id in enumerate(results['ids']):
                        item = {
                            'id': str(id),
                            'text': results.get('documents', [None])[i] or '',
                            'collection': collection_name
                        }
                        
                        # Add embedding if available
                        if 'embeddings' in results and i < len(results['embeddings']):
                            embedding = results['embeddings'][i]
                            item['embedding'] = embedding
                            # Add individual embedding dimensions as columns for atlas
                            for j, val in enumerate(embedding):
                                item[f'dim_{j}'] = val
                        
                        # Add metadata if available
                        if 'metadatas' in results and i < len(results['metadatas']):
                            metadata = results['metadatas'][i] or {}
                            for key, value in metadata.items():
                                if key not in ['id', 'text', 'embedding']:
                                    item[f'meta_{key}'] = value
                        
                        collection_data.append(item)
                        
                        if limit and len(collection_data) >= limit:
                            break
                    
                    logger.info(f"Exported {len(collection_data)} embeddings from ChromaDB collection {collection_name}")
                    return collection_data
                    
            except Exception as e:
                logger.debug(f"ChromaDB-style get failed for {collection_name}: {e}")
        
        # Strategy 3: Fallback - try using search with dummy query
        try:
            logger.info(f"Using search fallback for {collection_name}")
            # Use a very generic search to get all data
            search_results = await vector_engine.search(
                collection_name=collection_name,
                query_text="the",  # Use a common word instead of empty query
                limit=limit or 10000,
                with_vector=True
            )
            
            if search_results:
                for result in search_results:
                    if isinstance(result, ScoredResult):
                        item = {
                            'id': str(result.id),
                            'text': _extract_text_from_payload(result.payload),
                            'collection': collection_name,
                            'score': result.score
                        }
                        
                        # Add embedding if available
                        if hasattr(result, 'vector') and result.vector:
                            embedding = result.vector
                            item['embedding'] = embedding
                            # Add individual embedding dimensions
                            for j, val in enumerate(embedding):
                                item[f'dim_{j}'] = val
                        
                        # Add payload metadata
                        if result.payload:
                            for key, value in result.payload.items():
                                if key not in ['id', 'text', 'embedding']:
                                    item[f'meta_{key}'] = value
                        
                        collection_data.append(item)
                
                logger.info(f"Exported {len(collection_data)} embeddings using search fallback for {collection_name}")
                return collection_data
                
        except Exception as e:
            logger.debug(f"Search-based export failed for {collection_name}: {e}")
        
        logger.warning(f"Could not export embeddings from {collection_name}")
        return []
        
    except Exception as e:
        logger.error(f"Error getting embeddings from {collection_name}: {e}")
        return []


def _extract_text_from_payload(payload: Dict[str, Any]) -> str:
    """Extract text content from payload data."""
    if not payload:
        return ""
    
    # Common text field names
    text_fields = ['text', 'content', 'document', 'data', 'name', 'title']
    
    for field in text_fields:
        if field in payload and payload[field]:
            return str(payload[field])
    
    # If no standard text field found, try to find any string value
    for key, value in payload.items():
        if isinstance(value, str) and len(value.strip()) > 0:
            return value
    
    return ""
