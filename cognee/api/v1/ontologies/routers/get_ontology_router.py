from fastapi import APIRouter, File, Form, UploadFile, Depends, Request
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
        ontology_key: str = Form(...),
        ontology_file: UploadFile = File(...),
        description: Optional[str] = Form(None),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Upload a single ontology file for later use in cognify operations.

        ## Request Parameters
        - **ontology_key** (str): User-defined identifier for the ontology.
        - **ontology_file** (UploadFile): Single OWL format ontology file
        - **description** (Optional[str]): Optional description for the ontology.

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
            metadata = ontology_service.list_ontologies(user)
            return metadata
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    return router
