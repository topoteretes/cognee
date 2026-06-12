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


async def _import_cogx_archives(uploads, dataset_name: str, import_mode, user):
    """Import uploaded COGX archive tarballs (produced by ``cognee.push()``)."""
    import tempfile

    from cognee.modules.migration import COGXArchiveSource, import_memory_source
    from cognee.modules.migration.archive import unpack_archive
    from cognee.modules.migration.sources.base import IMPORT_MODES

    if import_mode and import_mode not in IMPORT_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown import_mode {import_mode!r}. Expected one of {IMPORT_MODES}.",
        )

    result = None
    try:
        for upload in uploads:
            with tempfile.TemporaryDirectory() as temporary_directory:
                archive_root = unpack_archive(upload.file, temporary_directory)
                source = COGXArchiveSource(archive_root, mode=import_mode or "preserve")
                result = await import_memory_source(source, dataset_name=dataset_name, user=user)
        if result is None:
            raise HTTPException(status_code=400, detail="No archive files were processed.")
        return jsonable_encoder(result.to_dict())
    except ValueError as error:
        logger.error("COGX archive import validation error: %s", error, exc_info=True)
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid COGX archive: {error}"},
        )
    except Exception as error:
        logger.error("COGX archive import error: %s", error, exc_info=True)
        return JSONResponse(
            status_code=409,
            content={"error": "An error occurred during COGX archive import."},
        )


def get_remember_router() -> APIRouter:
    router = APIRouter()

    @router.post("", response_model=dict)
    @log_usage(function_name="POST /v1/remember", log_type="api_endpoint")
    async def remember(
        data: List[UploadFile] = File(default=None),
        datasetName: Optional[str] = Form(default=None),
        datasetId: Union[UUID, Literal[""], None] = Form(default=None, examples=[""]),
        session_id: Optional[str] = Form(default=None, examples=[""]),
        node_set: Optional[List[str]] = Form(default=[""], example=[""]),
        run_in_background: Optional[bool] = Form(default=False),
        custom_prompt: Optional[str] = Form(default=""),
        chunk_size: Optional[int] = Form(default=4096),
        chunks_per_batch: Optional[int] = Form(default=36),
        ontology_key: Optional[List[str]] = Form(
            default=None,
            examples=[[]],
            description="Reference to one or more previously uploaded ontologies",
        ),
        graph_model: Optional[str] = Form(
            default=None,
            examples=[""],
            description="JSON-serialised graph model schema (same format as the cognify endpoint).",
        ),
        content_type: Optional[str] = Form(default=None, examples=[""]),
        import_mode: Optional[str] = Form(
            default=None,
            examples=[""],
            description=(
                "COGX archive imports only: 'preserve' (default), 'hybrid', or 're-derive'."
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
        - **node_set** (Optional[List[str]]): Node identifiers for graph organisation.
        - **run_in_background** (Optional[bool]): Run the cognify step asynchronously (default: False).
        - **custom_prompt** (Optional[str]): Custom prompt for entity extraction.
        - **chunk_size** (Optional[int]): Maximum tokens per chunk. Defaults to automatic
          model-based sizing.
        - **chunks_per_batch** (Optional[int]): Chunks per cognify batch.
        - **ontology_key** (Optional[List[str]]): Reference to one or more previously uploaded ontology files to use for knowledge graph construction.
        - **graph_model** (Optional[str]): JSON-serialised graph model schema (same dict format accepted by the cognify endpoint).

        Either datasetName or datasetId must be provided.

        ## Error Codes
        - **400 Bad Request**: Neither datasetId nor datasetName provided
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

        if content_type == "cogx-archive":
            if not data:
                raise HTTPException(
                    status_code=400,
                    detail="content_type 'cogx-archive' requires an uploaded archive file.",
                )
            if not datasetName:
                raise HTTPException(
                    status_code=400,
                    detail="datasetName must be provided for COGX archive imports.",
                )
            return await _import_cogx_archives(data, datasetName, import_mode, user)

        from cognee.api.v1.remember import remember as cognee_remember
        from cognee.api.v1.ontologies.ontologies import OntologyService
        from cognee.shared.graph_model_utils import graph_schema_to_graph_model

        try:
            config_to_use = None
            if ontology_key and ontology_key != [""]:
                ontology_service = OntologyService()
                ontology_contents = ontology_service.get_ontology_contents(ontology_key, user)

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

            graph_model_parsed = None
            if graph_model:
                try:
                    graph_model_schema = json.loads(graph_model)
                    graph_model_parsed = graph_schema_to_graph_model(graph_model_schema)
                except (json.JSONDecodeError, Exception) as parse_err:
                    logger.warning("remember: invalid graph_model JSON, ignoring: %s", parse_err)

            result = await cognee_remember(
                data,
                dataset_name=datasetName,
                session_id=session_id,
                user=user,
                dataset_id=datasetId if datasetId else None,
                node_set=node_set if node_set != [""] else None,
                run_in_background=run_in_background or False,
                custom_prompt=custom_prompt or None,
                chunk_size=chunk_size,
                chunks_per_batch=chunks_per_batch,
                content_type=content_type,
                **({"config": config_to_use} if config_to_use else {}),
                **({"graph_model": graph_model_parsed} if graph_model_parsed else {}),
            )

            return jsonable_encoder(result.to_dict())
        except ValueError as error:
            logger.error("Remember endpoint validation error: %s", error, exc_info=True)
            return JSONResponse(
                status_code=409,
                content={"error": "Invalid request data for remember operation."},
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
        session_id: Optional[str] = None
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
