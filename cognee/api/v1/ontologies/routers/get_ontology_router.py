from fastapi import APIRouter, File, Form, UploadFile, Depends, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional

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
        ontology_key: str = Form(...),
        ontology_file: UploadFile = File(...),
        description: Optional[str] = Form(None),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Upload an ontology file with a named key for later use in cognify operations.

        ## Request Parameters
        - **ontology_key** (str): User-defined identifier for the ontology
        - **ontology_file** (UploadFile): OWL format ontology file
        - **description** (Optional[str]): Optional description of the ontology

        ## Response
        Returns metadata about the uploaded ontology including key, filename, size, and upload timestamp.

        ## Error Codes
        - **400 Bad Request**: Invalid file format, duplicate key, file size exceeded
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
            result = await ontology_service.upload_ontology(
                ontology_key, ontology_file, user, description
            )
            return {
                "ontology_key": result.ontology_key,
                "filename": result.filename,
                "size_bytes": result.size_bytes,
                "uploaded_at": result.uploaded_at,
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
