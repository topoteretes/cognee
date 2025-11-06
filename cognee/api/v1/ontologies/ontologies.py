import os
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from dataclasses import dataclass


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
        self, ontology_key: str, file, user, description: Optional[str] = None
    ) -> OntologyMetadata:
        # Validate file format
        if not file.filename.lower().endswith(".owl"):
            raise ValueError("File must be in .owl format")

        user_dir = self._get_user_dir(str(user.id))
        metadata = self._load_metadata(user_dir)

        # Check for duplicate key
        if ontology_key in metadata:
            raise ValueError(f"Ontology key '{ontology_key}' already exists")

        # Read file content
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:  # 10MB limit
            raise ValueError("File size exceeds 10MB limit")

        # Save file
        file_path = user_dir / f"{ontology_key}.owl"
        with open(file_path, "wb") as f:
            f.write(content)

        # Update metadata
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

    def get_ontology_content(self, ontology_key: str, user) -> str:
        user_dir = self._get_user_dir(str(user.id))
        metadata = self._load_metadata(user_dir)

        if ontology_key not in metadata:
            raise ValueError(f"Ontology key '{ontology_key}' not found")

        file_path = user_dir / f"{ontology_key}.owl"
        if not file_path.exists():
            raise ValueError(f"Ontology file for key '{ontology_key}' not found")

        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    def list_ontologies(self, user) -> dict:
        user_dir = self._get_user_dir(str(user.id))
        return self._load_metadata(user_dir)
