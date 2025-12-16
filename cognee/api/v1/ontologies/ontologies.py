import os
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List
from dataclasses import dataclass
from fastapi import UploadFile


@dataclass
class OntologyMetadata:
    ontology_key: str
    filename: str
    size_bytes: int
    uploaded_at: str
    description: Optional[str] = None


class OntologyService:
    def __init__(self):
        pass

    @property
    def base_dir(self) -> Path:
        return Path(tempfile.gettempdir()) / "ontologies"

    def _get_user_dir(self, user_id: str) -> Path:
        user_dir = self.base_dir / str(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        return user_dir

    def _get_metadata_path(self, user_dir: Path) -> Path:
        return user_dir / "metadata.json"

    def _load_metadata(self, user_dir: Path) -> dict:
        metadata_path = self._get_metadata_path(user_dir)
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                return json.load(f)
        return {}

    def _save_metadata(self, user_dir: Path, metadata: dict):
        metadata_path = self._get_metadata_path(user_dir)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    async def upload_ontology(
        self, ontology_key: str, file: UploadFile, user, description: Optional[str] = None
    ) -> OntologyMetadata:
        if not file.filename:
            raise ValueError("File must have a filename")
        if not file.filename.lower().endswith(".owl"):
            raise ValueError("File must be in .owl format")

        user_dir = self._get_user_dir(str(user.id))
        metadata = self._load_metadata(user_dir)

        if ontology_key in metadata:
            raise ValueError(f"Ontology key '{ontology_key}' already exists")

        content = await file.read()

        file_path = user_dir / f"{ontology_key}.owl"
        with open(file_path, "wb") as f:
            f.write(content)

        ontology_metadata = {
            "filename": file.filename,
            "size_bytes": len(content),
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "description": description,
        }
        metadata[ontology_key] = ontology_metadata
        self._save_metadata(user_dir, metadata)

        return OntologyMetadata(
            ontology_key=ontology_key,
            filename=file.filename,
            size_bytes=len(content),
            uploaded_at=ontology_metadata["uploaded_at"],
            description=description,
        )

    async def upload_ontologies(
        self,
        ontology_key: List[str],
        files: List[UploadFile],
        user,
        descriptions: Optional[List[str]] = None,
    ) -> List[OntologyMetadata]:
        """
        Upload ontology files with their respective keys.

        Args:
            ontology_key: List of unique keys for each ontology
            files: List of UploadFile objects (same length as keys)
            user: Authenticated user
            descriptions: Optional list of descriptions for each file

        Returns:
            List of OntologyMetadata objects for uploaded files

        Raises:
            ValueError: If keys duplicate, file format invalid, or array lengths don't match
        """
        if len(ontology_key) != len(files):
            raise ValueError("Number of keys must match number of files")

        if len(set(ontology_key)) != len(ontology_key):
            raise ValueError("Duplicate ontology keys not allowed")

        results = []

        for i, (key, file) in enumerate(zip(ontology_key, files)):
            results.append(
                await self.upload_ontology(
                    ontology_key=key,
                    file=file,
                    user=user,
                    description=descriptions[i] if descriptions else None,
                )
            )
        return results

    def get_ontology_contents(self, ontology_key: List[str], user) -> List[str]:
        """
        Retrieve ontology content for one or more keys.

        Args:
            ontology_key: List of ontology keys to retrieve (can contain single item)
            user: Authenticated user

        Returns:
            List of ontology content strings

        Raises:
            ValueError: If any ontology key not found
        """
        user_dir = self._get_user_dir(str(user.id))
        metadata = self._load_metadata(user_dir)

        contents = []
        for key in ontology_key:
            if key not in metadata:
                raise ValueError(f"Ontology key '{key}' not found")

            file_path = user_dir / f"{key}.owl"
            if not file_path.exists():
                raise ValueError(f"Ontology file for key '{key}' not found")

            with open(file_path, "r", encoding="utf-8") as f:
                contents.append(f.read())
        return contents

    def list_ontologies(self, user) -> dict:
        user_dir = self._get_user_dir(str(user.id))
        return self._load_metadata(user_dir)
