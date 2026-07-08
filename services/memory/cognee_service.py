import os
import asyncio
import logging
from services.timeline.events import log_event

logger = logging.getLogger("cognee_service")

# Helper to import Cognee and handle settings
def configure_cognee():
    """Reads settings from env and applies them to environment before importing cognee."""
    # LLM config
    llm_prov = os.getenv("LLM_PROVIDER", "openai")
    if llm_prov == "groq":
        os.environ["LLM_PROVIDER"] = "openai"
        os.environ["LLM_API_KEY"] = os.getenv("GROQ_API_KEY") or os.getenv("LLM_API_KEY", "")
    elif llm_prov == "gemini":
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["LLM_API_KEY"] = os.getenv("GEMINI_API_KEY") or os.getenv("LLM_API_KEY", "")
    else:
        os.environ["LLM_PROVIDER"] = llm_prov
        os.environ["LLM_API_KEY"] = os.getenv("LLM_API_KEY", "")
        
    os.environ["LLM_MODEL"] = os.getenv("LLM_MODEL", "gpt-4o")
    
    # Embedding config
    os.environ["EMBEDDING_PROVIDER"] = os.getenv("EMBEDDING_PROVIDER", "openai")
    os.environ["EMBEDDING_MODEL"] = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    os.environ["EMBEDDING_DIMENSIONS"] = os.getenv("EMBEDDING_DIMENSIONS", "3072")
    os.environ["EMBEDDING_API_KEY"] = os.getenv("EMBEDDING_API_KEY", "")
    
    # Specific keys for LiteLLM
    if os.getenv("GEMINI_API_KEY"):
        os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
    if os.getenv("GROQ_API_KEY"):
        os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

    # DB config
    os.environ["RELATIONAL_DATABASE"] = os.getenv("RELATIONAL_DATABASE", "sqlite://./cognee.db")
    os.environ["GRAPH_DATABASE"] = os.getenv("GRAPH_DATABASE", "networkx")
    os.environ["VECTOR_DATABASE"] = os.getenv("VECTOR_DATABASE", "lancedb")

configure_cognee()

import cognee
from cognee import SearchType
from cognee.infrastructure.llm.config import get_llm_config

# Force the config object to see "openai" regardless of what Pydantic read from .env
_llm_config = get_llm_config()
if _llm_config.llm_provider == "groq":
    _llm_config.llm_provider = "openai"
    
async def with_retry(func, *args, **kwargs):
    max_retries = 2
    delay = 3
    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            error_str = str(e)
            if "InstructorRetryException" in error_str or "503" in error_str or "UNAVAILABLE" in error_str or "ServiceUnavailableError" in error_str:
                if attempt < max_retries:
                    logger.warning(f"Caught LLM busy error, retrying ({attempt+1}/{max_retries}) in {delay}s...")
                    await asyncio.sleep(delay)
                else:
                    raise Exception("LLM_SERVICE_BUSY")
            else:
                raise e

async def remember_data(data: str, dataset_name: str = "main_dataset", session_id: str = None) -> dict:
    """Ingests data into Cognee, runs cognify, and logs the event."""
    configure_cognee()
    
    # We log the initiation
    start_event = log_event("IngestionStarted", f"Started ingestion for dataset: {dataset_name}", {
        "dataset_name": dataset_name,
        "is_file": os.path.exists(data) if isinstance(data, str) and len(data) < 255 else False,
        "session_id": session_id
    })
    
    try:
        # Check if file path
        is_file = False
        display_name = "raw text block"
        if isinstance(data, str) and len(data) < 512 and os.path.exists(data):
            is_file = True
            display_name = os.path.basename(data)
            
        # Ingest and Cognify using Cognee's v2 remember API
        await with_retry(cognee.remember, data, dataset_name = dataset_name, session_id = session_id)
        
        # Log success event
        log_event("MemoryCreated", f"Successfully remembered {display_name} in {dataset_name}", {
            "dataset_name": dataset_name,
            "source": display_name,
            "is_file": is_file
        })
        
        return {"status": "success", "message": f"Successfully remembered {display_name}"}
        
    except Exception as e:
        logger.error(f"Error during remember: {e}", exc_info=True)
        log_event("IngestionFailed", f"Failed to ingest data: {str(e)}", {
            "dataset_name": dataset_name,
            "error": str(e)
        })
        raise e

async def recall_data(query: str, query_type_str: str = "GRAPH_COMPLETION", dataset_name: str = "main_dataset", session_id: str = None) -> list:
    """Retrieves context from Cognee based on query and query type."""
    configure_cognee()
    
    # Match query type
    search_type = SearchType.GRAPH_COMPLETION
    try:
        search_type = SearchType[query_type_str.upper()]
    except Exception:
        search_type = SearchType.GRAPH_COMPLETION
        
    # Log event
    log_event("RecallTriggered", f"Query: \"{query}\" using {query_type_str}", {
        "query": query,
        "query_type": query_type_str,
        "dataset_name": dataset_name,
        "session_id": session_id
    })
    
    try:
        # Search Cognee
        results = await with_retry(cognee.recall, query, query_type = search_type, session_id = session_id)
        
        # Format results to return
        formatted_results = []
        if isinstance(results, list):
            for r in results:
                # If Cognee returns objects with properties
                if hasattr(r, "text"):
                    formatted_results.append({"text": r.text, "metadata": getattr(r, "metadata", {})})
                elif isinstance(r, dict):
                    formatted_results.append(r)
                else:
                    formatted_results.append({"text": str(r)})
        else:
            if hasattr(results, "text"):
                formatted_results.append({"text": results.text, "metadata": getattr(results, "metadata", {})})
            else:
                formatted_results.append({"text": str(results)})
                
        # Log RecallFinished
        log_event("RecallFinished", f"Recall completed with {len(formatted_results)} results", {
            "query": query,
            "results_count": len(formatted_results)
        })
        
        return formatted_results
    except Exception as e:
        logger.error(f"Error during recall: {e}", exc_info=True)
        log_event("RecallFailed", f"Recall failed: {str(e)}", {
            "query": query,
            "error": str(e)
        })
        return [{"text": f"Error recalling memory: {str(e)}", "error": True}]

async def improve_memory() -> dict:
    """Triggers Cognee memory self-improvement/consolidation."""
    configure_cognee()
    log_event("MemoryImprovedStarted", "Running memory consolidation and alignment")
    
    try:
        # Call Cognee improve
        await with_retry(cognee.improve)
        
        log_event("MemoryImproved", "Memory consolidation completed successfully")
        return {"status": "success", "message": "Memory improved successfully"}
    except Exception as e:
        logger.error(f"Error during improve: {e}", exc_info=True)
        log_event("MemoryImprovedFailed", f"Consolidation failed: {str(e)}", {
            "error": str(e)
        })
        raise e

async def forget_memory(everything: bool = False, dataset_name: str = None) -> dict:
    """Deletes memory or datasets."""
    configure_cognee()
    
    desc = "Cleared all memories" if everything else f"Cleared dataset: {dataset_name}"
    log_event("MemoryForgottenStarted", desc)
    
    try:
        if everything:
            await with_retry(cognee.forget, everything = True)
        else:
            # Clear specific dataset
            await with_retry(cognee.forget, dataset_name = dataset_name)
            
        log_event("MemoryForgotten", desc, {
            "everything": everything,
            "dataset_name": dataset_name
        })
        return {"status": "success", "message": desc}
    except Exception as e:
        logger.error(f"Error during forget: {e}", exc_info=True)
        log_event("MemoryForgottenFailed", f"Forgetting failed: {str(e)}", {
            "error": str(e)
        })
        raise e
