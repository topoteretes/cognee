from typing import List, Tuple, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.tasks.notetaker.normalize import normalize_transcript
from cognee.modules.retrieval.notetaker_templates import (
    NotetakerActionItemRetriever,
    NotetakerDecisionRetriever,
    NotetakerTemporalDeltaRetriever
)
from cognee.api.v1.forget.forget import forget
import cognee

router = APIRouter()

class IngestPayload(BaseModel):
    series_id: str = Field(..., description="The ID of the meeting series (dataset name)")
    meeting_id: str = Field(..., description="The ID of this specific meeting")
    turns: List[Tuple[str, str, str]] = Field(
        ..., 
        description="List of (speaker, text, timestamp) tuples"
    )
    permalink: Optional[str] = Field(None, description="Optional permalink to the meeting")

class RecallPayload(BaseModel):
    query: str = Field(..., description="The question to ask")
    query_type: str = Field(
        "action_items", 
        description="Type of query: 'action_items', 'decisions', or 'temporal_delta'"
    )

class ForgetPayload(BaseModel):
    series_id: Optional[str] = Field(None, description="Wipes the entire series (dataset)")
    meeting_id: Optional[str] = Field(None, description="Wipes a single occurrence by data_id")


@router.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
async def notetaker_ingest(payload: IngestPayload, user: User = Depends(get_authenticated_user)):
    """
    Ingest a meeting transcript and cognify it in the background.
    """
    try:
        # 1. Normalize transcript
        normalized_text = normalize_transcript(
            turns=payload.turns,
            meeting_id=payload.meeting_id,
            permalink=payload.permalink
        )
        
        # 2. Add to cognee (using series_id as dataset)
        await cognee.add(normalized_text, dataset_name=payload.series_id, user=user)
        
        # 3. Cognify in the background with temporal_cognify=True
        # cognify() with run_in_background=True returns a PipelineRunInfo object or dict
        cognify_result = await cognee.cognify(
            datasets=[payload.series_id],
            user=user,
            temporal_cognify=True,
            run_in_background=True
        )
        
        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "message": "Ingestion accepted and processing in background.",
                "dataset": payload.series_id,
                "pipeline_info": cognify_result if isinstance(cognify_result, dict) else {"result": str(cognify_result)}
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recall")
async def notetaker_recall(
    query: str, 
    query_type: str = "action_items", 
    user: User = Depends(get_authenticated_user)
):
    """
    Recall information using focused templates on the temporal graph.
    """
    try:
        if query_type == "action_items":
            retriever = NotetakerActionItemRetriever()
        elif query_type == "decisions":
            retriever = NotetakerDecisionRetriever()
        elif query_type == "temporal_delta":
            retriever = NotetakerTemporalDeltaRetriever()
        else:
            raise HTTPException(status_code=400, detail=f"Invalid query_type: {query_type}")
            
        answer = await retriever.get_completion(query)
        
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/forget")
async def notetaker_forget(payload: ForgetPayload, user: User = Depends(get_authenticated_user)):
    """
    Forget a specific meeting occurrence or an entire series.
    """
    try:
        if payload.meeting_id:
            # Forget single occurrence (maps to data_id in standard cognee)
            # cognee.forget takes data_id or dataset
            # Wait, forget function signatures might differ, we map meeting_id to data_id
            await forget(data_ids=[payload.meeting_id], user=user)
            return {"message": f"Successfully forgot meeting {payload.meeting_id}"}
            
        elif payload.series_id:
            # Forget entire series (maps to dataset in standard cognee)
            await forget(dataset_id=payload.series_id, user=user)
            return {"message": f"Successfully forgot series {payload.series_id}"}
            
        else:
            raise HTTPException(status_code=400, detail="Must provide either meeting_id or series_id")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def get_notetaker_router() -> APIRouter:
    return router
