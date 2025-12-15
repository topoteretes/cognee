from fastapi import APIRouter, File, Form, UploadFile, Depends, HTTPException
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
        ontology_key: List[str] = Form(...),
        ontology_file: List[UploadFile] = File(...),
        descriptions: Optional[List[str]] = Form(None),
        user: User = Depends(get_authenticated_user),
    ):
        """
        Upload ontology files with their respective keys for later use in cognify operations.

        Supports both single and multiple file uploads:
        - Single file: ontology_key=["key"], ontology_file=[file]
        - Multiple files: ontology_key=["key1", "key2"], ontology_file=[file1, file2]

        ## Request Parameters
        - **ontology_key** (List[str]): Repeated field (e.g. ontology_key=foo&ontology_key=bar) of user-defined identifiers
        - **ontology_file** (List[UploadFile]): OWL format ontology files
        - **descriptions** (Optional[List[str]]): Repeated optional descriptions aligned with ontology_key

        ## Response
        Returns metadata about uploaded ontologies including keys, filenames, sizes, and upload timestamps.

        ## Error Codes
        - **400 Bad Request**: Invalid file format, duplicate keys, array length mismatches, file size exceeded
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
            results = await ontology_service.upload_ontologies(
                ontology_key, ontology_file, user, descriptions
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
                    for result in results
                ]
            }
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
