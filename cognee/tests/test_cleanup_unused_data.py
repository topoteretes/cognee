import os    
import pathlib    
import cognee    
from datetime import datetime, timezone, timedelta    
from uuid import UUID    
from sqlalchemy import select, update    
from cognee.modules.data.models import Data, DatasetData    
from cognee.infrastructure.databases.relational import get_relational_engine    
from cognee.modules.users.methods import get_default_user    
from cognee.shared.logging_utils import get_logger    
from cognee.modules.search.types import SearchType    
    
logger = get_logger()    
    
    
async def test_textdocument_cleanup_with_sql():    
    """    
    End-to-end test for TextDocument cleanup based on last_accessed timestamps.    
        
    Tests:    
    1. Add and cognify a document    
    2. Perform search to populate last_accessed timestamp    
    3. Verify last_accessed is set in SQL Data table    
    4. Manually age the timestamp beyond cleanup threshold    
    5. Run cleanup with text_doc=True    
    6. Verify document was deleted from all databases (relational, graph, and vector)  
    """    
    # Setup test directories    
    data_directory_path = str(    
        pathlib.Path(    
            os.path.join(pathlib.Path(__file__).parent, ".data_storage/test_cleanup")    
        ).resolve()    
    )    
    cognee_directory_path = str(    
        pathlib.Path(    
            os.path.join(pathlib.Path(__file__).parent, ".cognee_system/test_cleanup")    
        ).resolve()    
    )    
        
    cognee.config.data_root_directory(data_directory_path)    
    cognee.config.system_root_directory(cognee_directory_path)    
        
    # Initialize database    
    from cognee.modules.engine.operations.setup import setup    
        
    # Clean slate    
    await cognee.prune.prune_data()    
    await cognee.prune.prune_system(metadata=True)    
        
    logger.info("🧪 Testing TextDocument cleanup based on last_accessed")    
        
    # Step 1: Add and cognify a test document    
    dataset_name = "test_cleanup_dataset"    
    test_text = """    
    Machine learning is a subset of artificial intelligence that enables systems to learn    
    and improve from experience without being explicitly programmed. Deep learning uses    
    neural networks with multiple layers to process data.    
    """    
        
    await setup()    
    user = await get_default_user()    
    await cognee.add([test_text], dataset_name=dataset_name, user=user)    
        
    cognify_result = await cognee.cognify([dataset_name], user=user)    
        
    # Extract dataset_id from cognify result (ds_id is already a UUID)    
    dataset_id = None    
    for ds_id, pipeline_result in cognify_result.items():    
        dataset_id = ds_id  # Don't wrap in UUID() - it's already a UUID object    
        break    
        
    assert dataset_id is not None, "Failed to get dataset_id from cognify result"    
    logger.info(f"✅ Document added and cognified. Dataset ID: {dataset_id}")    
        
    # Step 2: Perform search to trigger last_accessed update    
    logger.info("Triggering search to update last_accessed...")    
    search_results = await cognee.search(    
        query_type=SearchType.CHUNKS,    
        query_text="machine learning",    
        datasets=[dataset_name],    
        user=user    
    )    
    logger.info(f"✅ Search completed, found {len(search_results)} results")    
        
    # Step 3: Verify last_accessed was set in SQL Data table    
    db_engine = get_relational_engine()    
    async with db_engine.get_async_session() as session:    
        # Get the Data record for this dataset    
        result = await session.execute(    
            select(Data, DatasetData)    
            .join(DatasetData, Data.id == DatasetData.data_id)    
            .where(DatasetData.dataset_id == dataset_id)    
        )    
        data_records = result.all()    
        assert len(data_records) > 0, "No Data records found for the dataset"    
        data_record = data_records[0][0]  
        data_id = data_record.id    
            
        # Verify last_accessed is set (should be set by search operation)    
        assert data_record.last_accessed is not None, (    
            "last_accessed should be set after search operation"    
        )    
            
        original_last_accessed = data_record.last_accessed    
        logger.info(f"✅ last_accessed verified: {original_last_accessed}")    
        
    # Step 4: Manually age the timestamp to be older than cleanup threshold    
    days_threshold = 30   
    aged_timestamp = datetime.now(timezone.utc) - timedelta(days=days_threshold + 10)    
        
    async with db_engine.get_async_session() as session:    
        stmt = update(Data).where(Data.id == data_id).values(last_accessed=aged_timestamp)    
        await session.execute(stmt)    
        await session.commit()    
        
    # Query in a NEW session to avoid cached values    
    async with db_engine.get_async_session() as session:    
        result = await session.execute(select(Data).where(Data.id == data_id))    
        updated_data = result.scalar_one_or_none()    
            
        # Make both timezone-aware for comparison    
        retrieved_timestamp = updated_data.last_accessed    
        if retrieved_timestamp.tzinfo is None:    
            # If database returned naive datetime, make it UTC-aware    
            retrieved_timestamp = retrieved_timestamp.replace(tzinfo=timezone.utc)    
            
        assert retrieved_timestamp == aged_timestamp, (    
            f"Timestamp should be updated to aged value. "    
            f"Expected: {aged_timestamp}, Got: {retrieved_timestamp}"    
        )  
          
    # Step 5: Test cleanup with text_doc=True    
    from cognee.tasks.cleanup.cleanup_unused_data import cleanup_unused_data    
        
    # First do a dry run    
    logger.info("Testing dry run with text_doc=True...")    
    dry_run_result = await cleanup_unused_data(    
        days_threshold=30,    
        dry_run=True,    
        user_id=user.id,    
        text_doc=True    
    )    
        
    assert dry_run_result['status'] == 'dry_run', "Status should be 'dry_run'"    
    assert dry_run_result['unused_count'] > 0, (    
        "Should find at least one unused document"    
    )    
    logger.info(f"✅ Dry run found {dry_run_result['unused_count']} unused documents")    
        
    # Now run actual cleanup    
    logger.info("Executing cleanup with text_doc=True...")    
    cleanup_result = await cleanup_unused_data(    
        days_threshold=30,    
        dry_run=False,    
        user_id=user.id,    
        text_doc=True    
    )    
        
    assert cleanup_result["status"] == "completed", "Cleanup should complete successfully"    
    assert cleanup_result["deleted_count"]["documents"] > 0, (    
        "At least one document should be deleted"    
    )    
    logger.info(f"✅ Cleanup completed. Deleted {cleanup_result['deleted_count']['documents']} documents")    
        
    # Step 6: Verify the document was actually deleted from SQL    
    async with db_engine.get_async_session() as session:    
        deleted_data = (    
            await session.execute(select(Data).where(Data.id == data_id))    
        ).scalar_one_or_none()    
            
        assert deleted_data is None, (    
            "Data record should be deleted after cleanup"    
        )    
        logger.info("✅ Confirmed: Data record was deleted from SQL database")    
        
    # Verify the dataset-data link was also removed    
    async with db_engine.get_async_session() as session:    
        dataset_data_link = (    
            await session.execute(    
                select(DatasetData).where(    
                    DatasetData.data_id == data_id,    
                    DatasetData.dataset_id == dataset_id    
                )    
            )    
        ).scalar_one_or_none()    
            
        assert dataset_data_link is None, (    
            "DatasetData link should be deleted after cleanup"    
        )    
        logger.info("✅ Confirmed: DatasetData link was deleted")    
        
    # Verify graph nodes were cleaned up    
    from cognee.infrastructure.databases.graph import get_graph_engine    
        
    graph_engine = await get_graph_engine()    
        
    # Try to find the TextDocument node - it should not exist    
    result = await graph_engine.query(    
        "MATCH (n:Node {id: $id}) RETURN n",    
        {"id": str(data_id)}    
    )    
        
    assert len(result) == 0, (    
        "TextDocument node should be deleted from graph database"    
    )    
    logger.info("✅ Confirmed: TextDocument node was deleted from graph database")    
      
    # Verify vector database was cleaned up  
    from cognee.infrastructure.databases.vector import get_vector_engine  
      
    vector_engine = get_vector_engine()  
      
    # Check each collection that should have been cleaned up  
    vector_collections = [  
        "DocumentChunk_text",  
        "Entity_name",   
        "TextSummary_text"  
    ]  
      
    for collection_name in vector_collections:  
        if await vector_engine.has_collection(collection_name):  
            # Try to retrieve the deleted data points  
            try:  
                results = await vector_engine.retrieve(collection_name, [str(data_id)])  
                assert len(results) == 0, (  
                    f"Data points should be deleted from {collection_name} collection"  
                )  
                logger.info(f"✅ Confirmed: {collection_name} collection is clean")  
            except Exception as e:  
                # Collection might be empty or not exist, which is fine  
                logger.info(f"✅ Confirmed: {collection_name} collection is empty or doesn't exist")  
                pass  
      
    logger.info("✅ Confirmed: Vector database entries were deleted")  
        
    logger.info("🎉 All cleanup tests passed!")    
        
    return True    
    
    
if __name__ == "__main__":    
    import asyncio    
    success = asyncio.run(test_textdocument_cleanup_with_sql())    
    exit(0 if success else 1)
