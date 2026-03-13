"""REST API router for cognee-skills.

Thin wrapper around cognee.cognee_skills.client.Skills — same interface
as the MCP tools, exposed over HTTP for the OpenClaw plugin and other
HTTP clients.
"""

from typing import List, Optional

from pydantic import BaseModel
from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user


# ---------------------------------------------------------------------------
# Request DTOs
# ---------------------------------------------------------------------------


class IngestPayload(BaseModel):
    skills_folder: str
    dataset_name: str = "skills"
    source_repo: str = ""
    skip_enrichment: bool = False
    node_set: str = "skills"


class UpsertPayload(BaseModel):
    skills_folder: str
    dataset_name: str = "skills"
    source_repo: str = ""
    node_set: str = "skills"


class ExecutePayload(BaseModel):
    skill_id: str
    task_text: str
    context: Optional[str] = None
    auto_observe: bool = True
    auto_evaluate: bool = True
    auto_amendify: bool = False
    amendify_min_runs: int = 3
    amendify_score_threshold: float = 0.5
    session_id: str = "default"
    node_set: str = "skills"


class ObservePayload(BaseModel):
    task_text: str
    selected_skill_id: str
    success_score: float
    session_id: str = "default"
    task_pattern_id: str = ""
    result_summary: str = ""
    candidate_skills: Optional[List[str]] = None
    feedback: float = 0.0
    error_type: str = ""
    error_message: str = ""
    latency_ms: int = 0


class InspectPayload(BaseModel):
    skill_id: str
    min_runs: int = 1
    score_threshold: float = 0.5
    node_set: str = "skills"


class PreviewAmendifyPayload(BaseModel):
    skill_id: str
    inspection_id: Optional[str] = None
    min_runs: int = 1
    score_threshold: float = 0.5
    node_set: str = "skills"


class AmendifyPayload(BaseModel):
    amendment_id: str
    write_to_disk: bool = False
    run_validation: bool = False
    validation_task_text: str = ""
    node_set: str = "skills"


class RollbackPayload(BaseModel):
    amendment_id: str
    write_to_disk: bool = False
    node_set: str = "skills"


class EvaluateAmendifyPayload(BaseModel):
    amendment_id: str
    node_set: str = "skills"


class AutoAmendifyPayload(BaseModel):
    skill_id: str
    min_runs: int = 1
    score_threshold: float = 0.5
    write_to_disk: bool = False
    run_validation: bool = False
    validation_task_text: str = ""
    node_set: str = "skills"


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def get_skills_router() -> APIRouter:
    router = APIRouter()

    def _client():
        from cognee.cognee_skills.client import skills

        return skills

    # -- Ingestion & management ---------------------------------------------

    @router.post("/ingest")
    async def ingest_skills(
        payload: IngestPayload, user: User = Depends(get_authenticated_user)
    ):
        from pathlib import Path

        folder = Path(payload.skills_folder).resolve()
        if not folder.is_dir():
            return JSONResponse(
                status_code=400,
                content={"error": f"Directory not found: {payload.skills_folder}"},
            )
        await _client().ingest(
            skills_folder=str(folder),
            dataset_name=payload.dataset_name,
            source_repo=payload.source_repo,
            skip_enrichment=payload.skip_enrichment,
            node_set=payload.node_set,
        )
        return {"status": "ok", "skills_folder": str(folder)}

    @router.post("/upsert")
    async def upsert_skills(
        payload: UpsertPayload, user: User = Depends(get_authenticated_user)
    ):
        from pathlib import Path

        folder = Path(payload.skills_folder).resolve()
        if not folder.is_dir():
            return JSONResponse(
                status_code=400,
                content={"error": f"Directory not found: {payload.skills_folder}"},
            )
        result = await _client().upsert(
            skills_folder=str(folder),
            dataset_name=payload.dataset_name,
            source_repo=payload.source_repo,
            node_set=payload.node_set,
        )
        return jsonable_encoder(result)

    @router.delete("/{skill_id}")
    async def remove_skill(skill_id: str, user: User = Depends(get_authenticated_user)):
        removed = await _client().remove(skill_id)
        if not removed:
            return JSONResponse(status_code=404, content={"error": f"Skill '{skill_id}' not found."})
        return {"status": "ok", "skill_id": skill_id}

    @router.get("")
    async def list_skills(
        node_set: str = "skills", user: User = Depends(get_authenticated_user)
    ):
        results = await _client().list(node_set=node_set)
        return jsonable_encoder(results)

    @router.get("/{skill_id}")
    async def load_skill(
        skill_id: str,
        node_set: str = "skills",
        user: User = Depends(get_authenticated_user),
    ):
        result = await _client().load(skill_id, node_set=node_set)
        if result is None:
            return JSONResponse(status_code=404, content={"error": f"Skill '{skill_id}' not found."})
        return jsonable_encoder(result)

    # -- Execution & observation ---------------------------------------------

    @router.post("/execute")
    async def execute_skill(
        payload: ExecutePayload, user: User = Depends(get_authenticated_user)
    ):
        result = await _client().execute(
            skill_id=payload.skill_id,
            task_text=payload.task_text,
            context=payload.context,
            auto_observe=payload.auto_observe,
            auto_evaluate=payload.auto_evaluate,
            auto_amendify=payload.auto_amendify,
            amendify_min_runs=payload.amendify_min_runs,
            amendify_score_threshold=payload.amendify_score_threshold,
            session_id=payload.session_id,
            node_set=payload.node_set,
        )
        return jsonable_encoder(result)

    @router.post("/observe")
    async def observe_skill_run(
        payload: ObservePayload, user: User = Depends(get_authenticated_user)
    ):
        result = await _client().observe(payload.model_dump())
        return jsonable_encoder(result)

    # -- Self-improvement ----------------------------------------------------

    @router.post("/inspect")
    async def inspect_skill(
        payload: InspectPayload, user: User = Depends(get_authenticated_user)
    ):
        result = await _client().inspect(
            skill_id=payload.skill_id,
            min_runs=payload.min_runs,
            score_threshold=payload.score_threshold,
            node_set=payload.node_set,
        )
        if result is None:
            return JSONResponse(
                status_code=200,
                content={"result": None, "message": "Insufficient failed runs for inspection."},
            )
        return jsonable_encoder(result)

    @router.post("/preview-amendify")
    async def preview_amendify_skill(
        payload: PreviewAmendifyPayload, user: User = Depends(get_authenticated_user)
    ):
        result = await _client().preview_amendify(
            skill_id=payload.skill_id,
            inspection_id=payload.inspection_id,
            min_runs=payload.min_runs,
            score_threshold=payload.score_threshold,
            node_set=payload.node_set,
        )
        if result is None:
            return JSONResponse(
                status_code=200,
                content={"result": None, "message": "No amendment proposed."},
            )
        return jsonable_encoder(result)

    @router.post("/amendify")
    async def amendify_skill(
        payload: AmendifyPayload, user: User = Depends(get_authenticated_user)
    ):
        result = await _client().amendify(
            amendment_id=payload.amendment_id,
            write_to_disk=payload.write_to_disk,
            validate=payload.run_validation,
            validation_task_text=payload.validation_task_text,
            node_set=payload.node_set,
        )
        return jsonable_encoder(result)

    @router.post("/rollback-amendify")
    async def rollback_amendify_skill(
        payload: RollbackPayload, user: User = Depends(get_authenticated_user)
    ):
        success = await _client().rollback_amendify(
            amendment_id=payload.amendment_id,
            write_to_disk=payload.write_to_disk,
            node_set=payload.node_set,
        )
        return {"success": success, "amendment_id": payload.amendment_id}

    @router.post("/evaluate-amendify")
    async def evaluate_amendify_skill(
        payload: EvaluateAmendifyPayload, user: User = Depends(get_authenticated_user)
    ):
        result = await _client().evaluate_amendify(
            amendment_id=payload.amendment_id,
            node_set=payload.node_set,
        )
        return jsonable_encoder(result)

    @router.post("/auto-amendify")
    async def auto_amendify_skill(
        payload: AutoAmendifyPayload, user: User = Depends(get_authenticated_user)
    ):
        result = await _client().auto_amendify(
            skill_id=payload.skill_id,
            min_runs=payload.min_runs,
            score_threshold=payload.score_threshold,
            write_to_disk=payload.write_to_disk,
            validate=payload.run_validation,
            validation_task_text=payload.validation_task_text,
            node_set=payload.node_set,
        )
        if result is None:
            return JSONResponse(
                status_code=200,
                content={"result": None, "message": "Insufficient failures to trigger amendify."},
            )
        return jsonable_encoder(result)

    return router
