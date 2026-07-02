import os
from typing import List, Tuple, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

import cognee
from cognee.api.v1.search import SearchType
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.tasks.notetaker.normalize import normalize_transcript

router = APIRouter()

# Focused answer prompts, resolved as absolute paths. These are passed to
# ``cognee.search(system_prompt_path=...)`` which reads them via
# ``read_query_prompt`` (``os.path.join`` on an absolute path returns the
# absolute path, so a non-default prompt location resolves correctly).
_PROMPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "modules",
    "retrieval",
    "prompts",
)
_RECALL_PROMPTS = {
    "action_items": os.path.join(_PROMPTS_DIR, "notetaker_action_items.txt"),
    "decisions": os.path.join(_PROMPTS_DIR, "notetaker_decisions.txt"),
    "temporal_delta": os.path.join(_PROMPTS_DIR, "notetaker_temporal_delta.txt"),
}


class IngestPayload(BaseModel):
    series_id: str = Field(..., description="The meeting series ID (used as the dataset name)")
    meeting_id: str = Field(..., description="The ID of this specific meeting occurrence")
    turns: List[Tuple[str, str, str]] = Field(
        ..., description="List of (speaker, text, timestamp) tuples"
    )
    permalink: Optional[str] = Field(None, description="Optional permalink to the meeting source")


class ForgetPayload(BaseModel):
    series_id: str = Field(..., description="The meeting series (dataset name)")
    data_id: Optional[str] = Field(
        None,
        description=(
            "Data UUID of a single occurrence to forget (as returned by /ingest). "
            "When omitted, the entire series dataset is forgotten."
        ),
    )


@router.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
async def notetaker_ingest(payload: IngestPayload, user: User = Depends(get_authenticated_user)):
    """Ingest a meeting transcript and cognify it (temporal) in the background.

    The series is the dataset, so recurring occurrences accumulate in one graph
    and temporal recall can span them. Returns the ``data_id`` of the ingested
    occurrence so it can be forgotten later.
    """
    try:
        normalized_text = normalize_transcript(
            turns=payload.turns,
            meeting_id=payload.meeting_id,
            permalink=payload.permalink,
        )

        add_info = await cognee.add(normalized_text, dataset_name=payload.series_id, user=user)

        # Surface the created data id(s) so a later /forget can target this exact
        # occurrence (cognee.forget needs the Data UUID, not the meeting_id).
        data_ids: List[str] = []
        payload_items = getattr(add_info, "payload", None)
        if isinstance(payload_items, list):
            for item in payload_items:
                item_id = getattr(item, "id", None)
                if item_id is not None:
                    data_ids.append(str(item_id))

        cognify_result = await cognee.cognify(
            datasets=[payload.series_id],
            user=user,
            temporal_cognify=True,
            run_in_background=True,
        )

        return JSONResponse(
            status_code=status.HTTP_202_ACCEPTED,
            content={
                "message": "Ingestion accepted; cognify running in background.",
                "series_id": payload.series_id,
                "meeting_id": payload.meeting_id,
                "data_ids": data_ids,
                "pipeline_run_id": str(getattr(cognify_result, "pipeline_run_id", "") or ""),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/recall")
async def notetaker_recall(
    series_id: str = Query(..., description="The meeting series (dataset) to recall from"),
    query: str = Query(..., description="The question to ask"),
    query_type: str = Query(
        "action_items", description="One of: 'action_items', 'decisions', 'temporal_delta'"
    ),
    user: User = Depends(get_authenticated_user),
):
    """Recall action items / decisions / "what changed" for a meeting series.

    Routes through ``cognee.search`` with ``SearchType.TEMPORAL`` scoped to the
    series dataset (permission-checked, dataset-bounded) and a focused answer
    prompt. ``include_references`` surfaces the citation-grounded source turns.
    """
    system_prompt_path = _RECALL_PROMPTS.get(query_type)
    if system_prompt_path is None:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid query_type '{query_type}'. Expected one of {list(_RECALL_PROMPTS)}.",
        )

    try:
        results = await cognee.search(
            query_text=query,
            query_type=SearchType.TEMPORAL,
            datasets=[series_id],
            system_prompt_path=system_prompt_path,
            user=user,
            include_references=True,
        )
        return {"series_id": series_id, "query_type": query_type, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/forget")
async def notetaker_forget(payload: ForgetPayload, user: User = Depends(get_authenticated_user)):
    """Forget a single occurrence (``data_id`` within the series) or the whole series."""
    try:
        if payload.data_id:
            try:
                data_uuid = UUID(payload.data_id)
            except ValueError:
                raise HTTPException(
                    status_code=400, detail=f"data_id is not a valid UUID: {payload.data_id}"
                )
            await cognee.forget(data_id=data_uuid, dataset=payload.series_id, user=user)
            return {
                "message": f"Forgot occurrence {payload.data_id} from series {payload.series_id}"
            }

        await cognee.forget(dataset=payload.series_id, user=user)
        return {"message": f"Forgot entire series {payload.series_id}"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def get_notetaker_router() -> APIRouter:
    return router
