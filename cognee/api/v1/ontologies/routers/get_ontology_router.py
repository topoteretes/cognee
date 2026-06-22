import asyncio

from fastapi import APIRouter, File, Form, Path, UploadFile, Depends, Request
from fastapi.responses import JSONResponse
from typing import Optional, List

from cognee.modules.users.models import User
from cognee.modules.users.methods import get_authenticated_user
from cognee.shared.utils import send_telemetry
from cognee import __version__ as cognee_version
from ..ontologies import OntologyService


def get_ontology_router() -> APIRouter:
    router = APIRouter()
    ontology_service = OntologyService()

    @router.post("", response_model=dict)
    async def upload_ontology(
        request: Request,
        ontology_key: str = Form(
            ...,
            examples=["medical_ontology"],
            description=(
                "Unique, user-defined identifier for this ontology. Reference it later via the "
                "ontology_key parameter of the cognify/remember endpoints."
            ),
        ),
        ontology_file: UploadFile = File(
            ...,
            description=(
                "Single ontology file in OWL (RDF/XML) format. The filename must end with .owl "
                "— other extensions are rejected with 400. Exactly one file per request."
            ),
        ),
        description: Optional[str] = Form(
            None,
            examples=["OWL ontology of medical conditions and treatments"],
            description=(
                "Optional human-readable description (plain string; values starting with "
                "'[' or '{' are rejected)."
            ),
        ),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Upload a single ontology file for later use in cognify operations.

        ## Request Parameters
        - **ontology_key** (str): Unique, user-defined identifier for the ontology (plain string — values starting with '[' or '{' are rejected; duplicate keys return 400). Use this key later as the `ontology_key` parameter in /api/v1/cognify or /api/v1/remember.
        - **ontology_file** (UploadFile): Single ontology file in OWL (RDF/XML) format; the filename must end with .owl.
        - **description** (Optional[str]): Optional description for the ontology (plain string; values starting with '[' or '{' are rejected).

        ## Response
        Returns metadata about the uploaded ontology including key, filename, size, and upload timestamp.

        ## Error Codes
        - **400 Bad Request**: Invalid file format, duplicate key, multiple files uploaded
        - **500 Internal Server Error**: File system or processing errors
        """
        send_telemetry(
            "Ontology Upload API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "POST /api/v1/ontologies",
                "cognee_version": cognee_version,
            },
        )

        try:
            # Enforce: exactly one uploaded file for "ontology_file"
            form = await request.form()
            uploaded_files = form.getlist("ontology_file")
            if len(uploaded_files) != 1:
                raise ValueError("Only one ontology_file is allowed")

            if ontology_key.strip().startswith(("[", "{")):
                raise ValueError("ontology_key must be a string")
            if description is not None and description.strip().startswith(("[", "{")):
                raise ValueError("description must be a string")

            result = await ontology_service.upload_ontology(
                ontology_key=ontology_key,
                file=ontology_file,
                user=user,
                description=description,
            )

            return {
                "uploaded_ontologies": [
                    {
                        "ontology_key": result.ontology_key,
                        "filename": result.filename,
                        "size_bytes": result.size_bytes,
                        "uploaded_at": result.uploaded_at,
                        "description": result.description,
                    }
                ]
            }
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    @router.delete("/{ontology_key}", response_model=dict)
    async def delete_ontology(
        ontology_key: str = Path(
            ...,
            examples=["medical_ontology"],
            description=(
                "Key of the ontology to delete, exactly as provided at upload time "
                "(see GET /api/v1/ontologies for available keys)."
            ),
        ),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Delete an uploaded ontology by key.

        ## Path Parameters
        - **ontology_key** (str): The key of the ontology to delete.

        ## Error Codes
        - **400 Bad Request**: Ontology key not found
        - **500 Internal Server Error**: File system errors
        """
        send_telemetry(
            "Ontology Delete API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "DELETE /api/v1/ontologies/{ontology_key}",
                "cognee_version": cognee_version,
            },
        )

        try:
            # delete_ontology performs blocking filesystem IO (stat/unlink); run
            # it off the event loop so this async route does not block.
            await asyncio.to_thread(
                ontology_service.delete_ontology, ontology_key=ontology_key, user=user
            )
            return {"status": "success", "ontology_key": ontology_key}
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    @router.get("", response_model=dict)
    async def list_ontologies(user: User = Depends(get_authenticated_user)):
        """
        List all uploaded ontologies for the authenticated user.

        ## Response
        Returns a dictionary mapping ontology keys to their metadata including filename, size, and upload timestamp.

        ## Error Codes
        - **500 Internal Server Error**: File system or processing errors
        """
        send_telemetry(
            "Ontology List API Endpoint Invoked",
            user.id,
            additional_properties={
                "endpoint": "GET /api/v1/ontologies",
                "cognee_version": cognee_version,
            },
        )

        try:
            # list_ontologies reads metadata from disk; run it off the event loop.
            metadata = await asyncio.to_thread(ontology_service.list_ontologies, user)
            return metadata
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    return router
