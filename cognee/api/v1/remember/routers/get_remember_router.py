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
from cognee.tasks.ingestion.data_item import (
    pair_labels_with_data,
    parse_external_metadata,
    parse_labels,
)
from cognee.shared.utils import send_telemetry
from cognee.shared.logging_utils import get_logger
from cognee.shared.usage_logger import log_usage
from cognee import __version__ as cognee_version
from cognee.exceptions import CogneeApiError

logger = get_logger()

# NOTE: Needed because of: https://github.com/fastapi/fastapi/discussions/14975
#       Once issue is resolved on Swagger side it can be removed.
UploadFile = Annotated[UF, WithJsonSchema({"type": "string", "format": "binary"})]

# Swagger UI prefills newly added array items from the ITEM-level example;
# without one it inserts the literal "string". An empty item example keeps
# "Add item" runnable (empty entries are filtered out server-side).
EmptyExampleStr = Annotated[str, WithJsonSchema({"type": "string", "example": ""})]


async def _import_cogx_archives(
    uploads,
    dataset_name,
    dataset_id,
    import_mode,
    user,
    run_in_background: bool = False,
):
    """Import uploaded COGX archive tarballs (produced by ``cognee.push()``)."""
    import tarfile
    import tempfile

    from cognee.modules.migration import COGXArchiveSource, import_memory_source
    from cognee.modules.migration.archive import unpack_archive
    from cognee.modules.migration.sources.base import IMPORT_MODES
    from cognee.modules.pipelines.layers.resolve_authorized_user_datasets import (
        resolve_authorized_user_datasets,
    )

    if import_mode and import_mode not in IMPORT_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown import_mode {import_mode!r}. Expected one of {IMPORT_MODES}.",
        )

    try:
        if not dataset_name:
            # The endpoint contract accepts datasetId in place of datasetName;
            # resolve the name the same way the main remember path does.
            _, authorized_datasets = await resolve_authorized_user_datasets(dataset_id, user)
            dataset_name = authorized_datasets[0].name

        results = []
        for upload in uploads:
            with tempfile.TemporaryDirectory() as temporary_directory:
                archive_root = unpack_archive(upload.file, temporary_directory)
                source = COGXArchiveSource(archive_root, mode=import_mode or "preserve")
                results.append(
                    await import_memory_source(
                        source,
                        dataset_name=dataset_name,
                        user=user,
                        run_in_background=run_in_background,
                    )
                )
        if not results:
            raise HTTPException(status_code=400, detail="No archive files were processed.")

        aggregate = results[-1].to_dict()
        aggregate["items_processed"] = sum(result.items_processed for result in results)
        items = [item for result in results for item in result.items]
        if items:
            aggregate["items"] = items
        return jsonable_encoder(aggregate)
    except HTTPException:
        raise
    except CogneeApiError:
        # Cognee errors (e.g. permission denied) carry their own status code
        # and actionable message; the global handler in cognee/api/client.py
        # returns them.
        raise
    except (ValueError, tarfile.TarError) as error:
        # Log the detail server-side; the response stays generic so exception
        # text / stack frames are not exposed to the caller (CodeQL py/stack-trace-exposure).
        logger.error("COGX archive import validation error: %s", error, exc_info=True)
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid COGX archive."},
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
        labels: Optional[str] = Form(
            default=None,
            examples=[""],
            description=(
                'Per-file labels, e.g. ["finance", "people", ""] — the Nth label applies '
                "to the Nth uploaded file, one entry per file, an empty entry skips that "
                'file. The comma-separated form "finance,people," is accepted '
                "equivalently (it is what Swagger UI sends when you type a JSON array "
                "here), so labels cannot contain commas unless the client sends real "
                "JSON. Stored on each file's data record and returned when listing "
                "dataset data. Only supported for normal ingestion — rejected when combined "
                "with session_id or content_type."
            ),
        ),
        external_metadata: Optional[str] = Form(
            default=None,
            examples=[""],
            description=(
                "JSON array of per-file metadata objects, e.g. "
                '[{"source": "crm", "ticket": 42}, null]. Paired positionally like labels: '
                "the Nth entry applies to the Nth uploaded file (null or {} skips that "
                "file), and one entry per file is required when any is given. Merged into "
                "the file's stored external_metadata (your keys win over loader-derived "
                "ones; 'node_set' is reserved). Only supported for normal ingestion — "
                "rejected when combined with session_id or content_type."
            ),
        ),
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
            default=None,
            examples=[None],
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
                'e.g. {"title": "CompanyGraph", "type": "object", "properties": {...}}. '
                "Must include a top-level 'title' key. Leave empty to use the default "
                "KnowledgeGraph model — a restrictive schema here can produce an empty graph. "
                "Invalid JSON or an unconvertible schema is rejected with 400."
            ),
        ),
        content_type: Optional[str] = Form(
            default=None,
            examples=[""],
            description=(
                "Set to 'skills' to ingest SKILL.md files as dataset-scoped Skill nodes, "
                "or 'code' to index whole code repositories (pass them via 'repositories') "
                "as an architectural code graph through the enola-backed pipeline. "
                "Leave empty for normal ingestion."
            ),
        ),
        import_mode: Optional[str] = Form(
            default=None,
            examples=[""],
            description=(
                "COGX archive imports only: 'preserve' (default), 'hybrid', or 're-derive'."
            ),
        ),
        skills_text: Optional[str] = Form(
            default=None,
            examples=[""],
            description=(
                "content_type='skills' only: inline SKILL.md markdown to ingest without a file "
                "upload (no-code path). When set and no files are uploaded, it is written to a "
                "temporary SKILL.md and ingested via the normal skills pipeline. Pair with "
                "skill_name to control the resulting skill name."
            ),
        ),
        skill_name: Optional[str] = Form(
            default=None,
            examples=[""],
            description=(
                "content_type='skills' + skills_text only: name/slug for the inline skill "
                "(defaults to 'skill')."
            ),
        ),
        repositories: Optional[List[EmptyExampleStr]] = Form(
            default=None,
            examples=[None],
            description=(
                "content_type='code' only: repository specs to index — remote git URLs "
                "(cloned server-side, shallow) or local directory paths on the server's "
                "filesystem (requires ACCEPT_LOCAL_FILE_PATH; useful when the server "
                "shares the caller's filesystem). One code graph is built per entry. "
                "Combine with run_in_background=true for large repositories and poll "
                "GET /v1/datasets/status?pipeline=code_graph_pipeline."
            ),
        ),
        index_vectors: Optional[bool] = Form(
            default=False,
            description=(
                "content_type='code' only: also embed the extracted code facts so "
                "semantic/completion retrievers can see them (requires an embedding "
                "provider). Default false — the code graph pipeline is deterministic "
                "and makes no LLM or embedding calls, and SearchType.CODE uses graph "
                "indexes only."
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
        - **labels** (Optional[str]): JSON array of per-file labels, e.g.
          ["finance", "people", ""], paired positionally with the uploaded files (one
          entry per file; an empty entry skips that file). Stored on each file's data
          record. Normal ingestion only — rejected with session_id or content_type.
        - **external_metadata** (Optional[str]): JSON array of per-file metadata objects,
          e.g. [{"source": "crm"}, null], paired positionally with the uploaded files
          (one entry per file; null or {} skips that file). Merged into each file's
          stored external_metadata. Normal ingestion only — rejected with session_id
          or content_type.
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
          Skill nodes, or "code" to index whole repositories (see repositories);
          omit for normal ingestion.
        - **repositories** (Optional[List[str]]): content_type="code" only — git URLs or
          server-local repo paths to index as code graphs, one graph per entry. Poll
          progress via GET /v1/datasets/status?pipeline=code_graph_pipeline.
        - **index_vectors** (Optional[bool]): content_type="code" only — also embed the
          extracted code facts for semantic retrievers (default false, no LLM/embedding
          calls otherwise).

        Either datasetName or datasetId must be provided.
        - **import_mode** (Optional[str]): COGX archive imports only: 'preserve' (default),
          'hybrid', or 're-derive'.
        - **skill_name** (Optional[str]): content_type='skills' + skills_text only: name/slug for
          the inline skill (defaults to 'skill').
        - **skills_text** (Optional[str]): content_type='skills' only: inline SKILL.md markdown to
          ingest without a file upload (no-code path). When set and no files are uploaded, it is
          written to a temporary SKILL.md and ingested via the normal skills pipeline. Pair with
          skill_name to control the resulting skill name.

        ## Error Codes
        - **400 Bad Request**: Neither datasetId nor datasetName provided, unsupported
          content_type, invalid graph_model JSON/schema, or invalid code-ingestion
          combination (missing repositories, file uploads or session_id with
          content_type="code", repositories/index_vectors without it, or local repo
          paths while ACCEPT_LOCAL_FILE_PATH=false)
        - **409 Conflict**: Error during processing
        """
        send_telemetry(
            "Remember API Endpoint Invoked",
            user,
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

        # Invalid JSON raises a CogneeApiError (400) via the global handler.
        parsed_labels = parse_labels(labels)
        parsed_metadata = parse_external_metadata(external_metadata)

        # Labels and metadata live on the Data records that normal add+cognify
        # ingestion creates. The session-cache, skills, and archive paths never
        # create those records, so they would be silently dropped — reject
        # instead.
        if (any(parsed_labels or []) or any(entry for entry in (parsed_metadata or []))) and (
            session_id or content_type
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    "labels and external_metadata are only supported for normal ingestion — "
                    "remove session_id and content_type to use them."
                ),
            )

        # Labels and metadata ride on DataItems, which ingestion unwraps to
        # store them on each file's Data record. A count mismatch raises a
        # CogneeApiError (400), returned by the global handler.
        data = pair_labels_with_data(data, parsed_labels, parsed_metadata)

        if content_type == "cogx-archive":
            if not data:
                raise HTTPException(
                    status_code=400,
                    detail="content_type 'cogx-archive' requires an uploaded archive file.",
                )
            return await _import_cogx_archives(
                data,
                datasetName,
                datasetId if datasetId else None,
                import_mode,
                user,
                run_in_background=run_in_background or False,
            )

        if content_type and content_type not in ("skills", "code", "cogx-archive"):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported content_type '{content_type}'. "
                    "Use 'skills', 'code', 'cogx-archive', or leave it empty for "
                    "normal ingestion."
                ),
            )

        # Drop empty entries — Swagger UI submits untouched array items as "".
        repo_specs = [spec.strip() for spec in (repositories or []) if spec and spec.strip()]

        if repo_specs and content_type != "code":
            raise HTTPException(
                status_code=400,
                detail="repositories is only supported with content_type='code'.",
            )
        if index_vectors and content_type != "code":
            raise HTTPException(
                status_code=400,
                detail="index_vectors is only supported with content_type='code'.",
            )

        if content_type == "code":
            if session_id:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "session_id is not applicable to content_type='code'; code graphs "
                        "are stored in the permanent graph, not a session cache."
                    ),
                )
            if data:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "content_type='code' does not accept file uploads — pass repository "
                        "paths or git URLs via 'repositories'. To ingest individual code "
                        "files, upload them under their real filename without content_type."
                    ),
                )
            if not repo_specs:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "content_type='code' requires at least one repository path or "
                        "git URL in 'repositories'."
                    ),
                )

            from cognee.tasks.code_graph.resolve_repo import is_remote_repo
            from cognee.tasks.ingestion.save_data_item_to_storage import (
                settings as save_data_settings,
            )

            # Local paths are read from the server's own filesystem. That is the
            # intended setup for a local server sharing the caller's checkout, but
            # a server that disables local file ingestion must not hand out a
            # read-any-directory primitive through this route either.
            if not save_data_settings.accept_local_file_path and any(
                not is_remote_repo(spec) for spec in repo_specs
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Local repository paths are disabled on this server "
                        "(ACCEPT_LOCAL_FILE_PATH=false) — pass a git URL instead."
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
                # For code, the payload is the repo specs — there are no uploads.
                repo_specs if content_type == "code" else data,
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
                skills_text=skills_text or None,
                skill_name=skill_name or None,
                # index_vectors may only reach remember() for code — it raises
                # for any other content_type, including None.
                **({"index_vectors": bool(index_vectors)} if content_type == "code" else {}),
                **({"config": config_to_use} if config_to_use else {}),
                **({"graph_model": graph_model_parsed} if graph_model_parsed else {}),
                # HTTP contract: an errored blocking run is reported as the 409
                # body below, not as an exception.
                raise_on_error=False,
            )

            # A blocking run that ended errored must not look like a success
            # to status-code-checking clients.
            if result.status == "errored":
                return JSONResponse(
                    status_code=409,
                    content=jsonable_encoder(result.to_dict()),
                )

            return jsonable_encoder(result.to_dict())
        except CogneeApiError:
            # Cognee errors carry their own status code and actionable message;
            # the global handler in cognee/api/client.py returns them.
            raise
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
                content={"error": f"An error occurred during remember: {error}"},
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
        dataset_id: Optional[UUID] = Field(
            default=None,
            description=(
                "UUID of an existing writable dataset. Takes precedence over dataset_name "
                "and is required to target a shared dataset by ID."
            ),
        )
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

        ## Request Parameters
        - **dataset_id** (Optional[UUID]): UUID of an existing writable dataset. Takes precedence
          over dataset_name and is required to target a shared dataset by ID.
        - **dataset_name** (str): Name of the target dataset. Defaults to 'main_dataset'.
        - **entry** (Union[QAEntry, TraceEntry, FeedbackEntry, SkillRunEntry]): Typed memory
          entry (qa, trace, feedback, or skill_run) to store, dispatched by its type field.
        - **session_id** (Optional[str]): Required for qa/trace/feedback entries; optional for
          skill_run entries.
        - **skill_improvement** (Optional[dict]): Skill improvement details forwarded to
          remember when recording a skill run.

        ## Response
        The returned ``RememberResult`` includes ``entry_type`` and
        ``entry_id`` — the ``qa_id``/``trace_id`` returned by the cache
        (or the ``qa_id`` a feedback was attached to). Use this to chain
        feedback to a freshly stored QA.
        """
        send_telemetry(
            "Remember Entry API Endpoint Invoked",
            user,
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
                dataset_id=payload.dataset_id,
                session_id=payload.session_id,
                user=user,
                skill_improvement=payload.skill_improvement,
            )
            return jsonable_encoder(result.to_dict())
        except ValueError as error:
            # Known validation errors: missing session_id, user not found, etc.
            logger.warning("Remember entry validation failed: %s", error)
            return JSONResponse(status_code=400, content={"error": "Invalid remember request."})
        except RuntimeError as error:
            # Session cache unavailable
            logger.error("Remember entry runtime failure: %s", error)
            return JSONResponse(
                status_code=503,
                content={"error": "Remember service temporarily unavailable."},
            )
        except CogneeApiError:
            # Cognee errors carry their own status code and actionable message;
            # the global handler in cognee/api/client.py returns them.
            raise
        except Exception as error:
            logger.error("Remember entry endpoint error: %s", error, exc_info=True)
            return JSONResponse(
                status_code=409,
                content={"error": "An error occurred during remember."},
            )

    return router
