import json
from uuid import UUID

from fastapi import APIRouter, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi import Form, File, UploadFile as UF, Depends
from typing import List, Optional, Union, Literal, Annotated
from pydantic import BaseModel, Field, WithJsonSchema

from cognee.memory import QAEntry, TraceEntry, FeedbackEntry, SkillRunEntry
from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.shared.utils import send_telemetry
from cognee.shared.logging_utils import get_logger
from cognee.shared.usage_logger import log_usage
from cognee import __version__ as cognee_version

logger = get_logger()

# NOTE: Needed because of: https://github.com/fastapi/fastapi/discussions/14975
#       Once issue is resolved on Swagger side it can be removed.
UploadFile = Annotated[UF, WithJsonSchema({"type": "string", "format": "binary"})]

# Swagger UI prefills newly added array items from the ITEM-level example;
# without one it inserts the literal "string". An empty item example keeps
# "Add item" runnable (empty entries are filtered out server-side).
EmptyExampleStr = Annotated[str, WithJsonSchema({"type": "string", "example": ""})]


def get_remember_router() -> APIRouter:
    router = APIRouter()

    @router.post("", response_model=dict)
    @log_usage(function_name="POST /v1/remember", log_type="api_endpoint")
    async def remember(
        data: List[UploadFile] = File(default=None),
        datasetName: Optional[str] = Form(
            default=None,
            examples=["default_dataset"],
            description=(
                "Name of the target dataset (created if it does not exist). "
                "Required unless datasetId is provided."
            ),
        ),
        datasetId: Union[UUID, Literal[""], None] = Form(default=None, examples=[""]),
        # examples=[""] keeps Swagger try-it-out runnable: without an example,
        # Swagger UI auto-generates the literal "string" and submits it.
        session_id: Optional[str] = Form(
            default=None,
            examples=[""],
            description=(
                "Session to attribute this memory to (e.g. claude-code-1718000000). "
                "When set, the data is stored in the session cache (and bridged into the "
                "permanent graph in the background) and the session appears in the sessions "
                "dashboard. Leave empty for a direct add+cognify."
            ),
        ),
        node_set: Optional[List[EmptyExampleStr]] = Form(
            default=[""],
            description=(
                "Tags the ingested data with named node sets (e.g. per-agent or per-project "
                "groups). Extracted graph nodes are linked to these sets, and recall/search "
                "can later be restricted to them via their node_name parameter. Leave empty "
                "to skip tagging."
            ),
        ),
        run_in_background: Optional[bool] = Form(
            default=False,
            description=(
                "If true, the request returns immediately (status 'running' with a "
                "pipeline_run_id) while ingestion and graph building continue server-side — "
                "poll GET /v1/datasets/status to track completion. If false, the request "
                "blocks until the knowledge graph is fully built, which can take minutes "
                "for large files."
            ),
        ),
        custom_prompt: Optional[str] = Form(
            default="",
            description=(
                "Replaces the default entity-extraction prompt used during graph building. "
                "Use it to steer which entities and relationships get extracted (e.g. focus "
                "on technical concepts, people, or contracts). Leave empty for the default "
                "prompt."
            ),
        ),
        chunk_size: Optional[int] = Form(
            default=4096,
            description=(
                "Maximum tokens per text chunk during ingestion (default: 4096). Each chunk "
                "is processed by the LLM separately for entity extraction: larger chunks give "
                "more context per extraction but fewer, coarser passes; smaller chunks give "
                "finer-grained extraction at higher LLM cost."
            ),
        ),
        chunks_per_batch: Optional[int] = Form(
            default=36,
            description=(
                "Number of chunks processed per cognify task batch (default: 36). Controls "
                "ingestion parallelism/throughput; rarely needs changing."
            ),
        ),
        ontology_key: Optional[List[EmptyExampleStr]] = Form(
            default=None,
            examples=[[]],
            description=(
                "Keys of previously uploaded ontologies (see /v1/ontologies) to ground "
                "entity extraction. Leave empty to ingest without an ontology."
            ),
        ),
        graph_model: Optional[str] = Form(
            default=None,
            examples=[""],
            description=(
                "JSON-serialised graph model schema (same format as the cognify endpoint), "
                "e.g. {\"title\": \"CompanyGraph\", \"type\": \"object\", \"properties\": {...}}. "
                "Must include a top-level 'title' key. Leave empty to use the default "
                "KnowledgeGraph model — a restrictive schema here can produce an empty graph. "
                "Invalid JSON or an unconvertible schema is rejected with 400."
            ),
        ),
        content_type: Optional[str] = Form(
            default=None,
            examples=[""],
            description=(
                "Set to 'skills' to ingest SKILL.md files as dataset-scoped Skill nodes. "
                "Only supported value: 'skills'; leave empty for normal ingestion."
            ),
        ),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Ingest data and build the knowledge graph in a single call.

        This endpoint combines the add and cognify steps. Data is ingested
        first, then automatically processed into a structured knowledge graph.

        ## Request Parameters
        - **data** (List[UploadFile]): Files to upload and process.
        - **datasetName** (Optional[str]): Name of the target dataset.
        - **datasetId** (Optional[UUID]): UUID of an existing dataset.
        - **session_id** (Optional[str]): Session to attribute this memory to. When set,
          data is stored in the session cache and bridged into the permanent graph in the
          background; the session is tracked in the sessions dashboard. When omitted,
          data is ingested directly via add + cognify.
        - **node_set** (Optional[List[str]]): Node identifiers for graph organisation.
        - **run_in_background** (Optional[bool]): Run the cognify step asynchronously (default: False).
        - **custom_prompt** (Optional[str]): Custom prompt for entity extraction.
        - **chunk_size** (Optional[int]): Maximum tokens per chunk (default: 4096).
        - **chunks_per_batch** (Optional[int]): Chunks per cognify batch.
        - **ontology_key** (Optional[List[str]]): Reference to one or more previously uploaded ontology files to use for knowledge graph construction.
        - **graph_model** (Optional[str]): JSON-serialised graph model schema (same dict format accepted by the cognify endpoint).
        - **content_type** (Optional[str]): Set to "skills" to ingest SKILL.md files as
          Skill nodes; omit for normal ingestion.

        Either datasetName or datasetId must be provided.

        ## Error Codes
        - **400 Bad Request**: Neither datasetId nor datasetName provided, unsupported
          content_type, or invalid graph_model JSON/schema
        - **409 Conflict**: Error during processing
        """
        send_telemetry(
            "Remember API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "POST /v1/remember",
                "node_set": node_set,
                "cognee_version": cognee_version,
            },
        )

        if not datasetId and not datasetName:
            raise HTTPException(
                status_code=400,
                detail="Either datasetId or datasetName must be provided.",
            )

        if content_type and content_type != "skills":
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported content_type '{content_type}'. "
                    "Use 'skills' or leave it empty for normal ingestion."
                ),
            )

        from cognee.api.v1.remember import remember as cognee_remember
        from cognee.api.v1.ontologies.ontologies import OntologyService
        from cognee.shared.graph_model_utils import graph_schema_to_graph_model

        # Validate graph_model before the generic try/except so failures
        # surface as a clear 400 instead of being swallowed into a 409.
        graph_model_parsed = None
        if graph_model:
            try:
                graph_model_schema = json.loads(graph_model)
            except json.JSONDecodeError as parse_err:
                raise HTTPException(
                    status_code=400,
                    detail=f"graph_model is not valid JSON: {parse_err}",
                )
            try:
                graph_model_parsed = graph_schema_to_graph_model(graph_model_schema)
            except Exception as parse_err:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"graph_model could not be converted to a graph model schema: {parse_err}. "
                        "Expected the same dict format as the cognify endpoint, "
                        "including a top-level 'title' key."
                    ),
                )

        try:
            config_to_use = None
            # Drop empty entries — Swagger UI submits untouched array items as "".
            ontology_keys = [key for key in (ontology_key or []) if key]
            if ontology_keys:
                ontology_service = OntologyService()
                ontology_contents = ontology_service.get_ontology_contents(ontology_keys, user)

                from cognee.modules.ontology.ontology_config import Config
                from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import (
                    RDFLibOntologyResolver,
                )
                from io import StringIO

                ontology_streams = [StringIO(content) for content in ontology_contents]
                config_to_use: Config = {
                    "ontology_config": {
                        "ontology_resolver": RDFLibOntologyResolver(ontology_file=ontology_streams)
                    }
                }


            result = await cognee_remember(
                data,
                dataset_name=datasetName,
                session_id=session_id or None,
                user=user,
                dataset_id=datasetId if datasetId else None,
                node_set=[tag for tag in (node_set or []) if tag] or None,
                run_in_background=run_in_background or False,
                custom_prompt=custom_prompt or None,
                chunk_size=chunk_size,
                chunks_per_batch=chunks_per_batch,
                # Swagger UI submits every rendered form field, so an untouched
                # content_type arrives as "" — treat it as omitted.
                content_type=content_type or None,
                **({"config": config_to_use} if config_to_use else {}),
                **({"graph_model": graph_model_parsed} if graph_model_parsed else {}),
            )

            return jsonable_encoder(result.to_dict())
        except ValueError as error:
            logger.error("Remember endpoint validation error: %s", error, exc_info=True)
            return JSONResponse(
                status_code=409,
                content={"error": f"Invalid request data for remember operation: {error}"},
            )
        except Exception as error:
            logger.error("Remember endpoint error: %s", error, exc_info=True)
            return JSONResponse(
                status_code=409,
                content={"error": "An error occurred during remember."},
            )

    class RememberEntryRequest(BaseModel):
        """JSON body for the typed-entry remember endpoint.

        ``entry`` is a discriminated union — set ``type`` to ``qa``,
        ``trace``, ``feedback``, or ``skill_run`` and include the
        corresponding fields.
        """

        entry: Annotated[
            Union[QAEntry, TraceEntry, FeedbackEntry, SkillRunEntry],
            Field(discriminator="type"),
        ]
        dataset_name: str = "main_dataset"
        session_id: Optional[str] = Field(
            default=None,
            examples=["claude-code-1718000000"],
            description="Required for qa/trace/feedback entries; optional for skill_run entries.",
        )
        skill_improvement: Optional[dict] = None

    @router.post("/entry", response_model=dict)
    @log_usage(function_name="POST /v1/remember/entry", log_type="api_endpoint")
    async def remember_entry(
        payload: RememberEntryRequest,
        user: User = Depends(get_authenticated_user),
    ):
        """Store a typed memory entry in the session cache.

        Accepts a discriminated union of ``QAEntry``, ``TraceEntry``,
        ``FeedbackEntry``, or ``SkillRunEntry`` and dispatches to the
        matching ``remember`` path. Session-backed entries require
        ``session_id``; ``SkillRunEntry`` can persist with or without one.

        ## Response
        The returned ``RememberResult`` includes ``entry_type`` and
        ``entry_id`` — the ``qa_id``/``trace_id`` returned by the cache
        (or the ``qa_id`` a feedback was attached to). Use this to chain
        feedback to a freshly stored QA.
        """
        send_telemetry(
            "Remember Entry API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "POST /v1/remember/entry",
                "entry_type": payload.entry.type,
                "cognee_version": cognee_version,
            },
        )

        from cognee.api.v1.remember import remember as cognee_remember

        try:
            result = await cognee_remember(
                payload.entry,
                dataset_name=payload.dataset_name,
                session_id=payload.session_id,
                user=user,
                skill_improvement=payload.skill_improvement,
            )
            return jsonable_encoder(result.to_dict())
        except ValueError as error:
            # Known validation errors: missing session_id, user not found, etc.
            return JSONResponse(status_code=400, content={"error": str(error)})
        except RuntimeError as error:
            # Session cache unavailable
            return JSONResponse(status_code=503, content={"error": str(error)})
        except Exception as error:
            logger.error("Remember entry endpoint error: %s", error, exc_info=True)
            return JSONResponse(
                status_code=409,
                content={"error": "An error occurred during remember."},
            )

    return router
