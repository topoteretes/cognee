import hashlib
import json
import os
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List
from dataclasses import dataclass
from fastapi import UploadFile

from cognee.base_config import get_base_config

_ONTOLOGY_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class DuplicateOntologyKeyError(ValueError):
    """Raised when an ontology key already exists for the user.

    A dedicated type lets the router surface a safe, literal client message
    without echoing the exception text (avoids stack-trace exposure).
    """


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
        base_config = get_base_config()
        return base_config.data_root_directory

    def _get_user_dir(self, user_id: str) -> Path:
        base_dir = os.path.normpath(os.path.abspath(os.path.expanduser(os.fspath(self.base_dir))))
        user_dir = os.path.normpath(os.path.abspath(os.path.join(base_dir, str(user_id))))
        base_prefix = base_dir if base_dir.endswith(os.sep) else f"{base_dir}{os.sep}"
        if user_dir != base_dir and not user_dir.startswith(base_prefix):
            raise ValueError("Invalid user id")
        user_dir = Path(user_dir)
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

    def _validate_ontology_key(self, ontology_key: str) -> str:
        normalized_key = ontology_key.strip()
        if not _ONTOLOGY_KEY_RE.fullmatch(normalized_key):
            raise ValueError("Invalid ontology key")
        return normalized_key

    def _get_ontology_path(self, user_dir: Path, ontology_key: str) -> Path:
        normalized_key = self._validate_ontology_key(ontology_key)
        storage_name = hashlib.blake2s(normalized_key.encode("utf-8"), digest_size=16).hexdigest()
        return user_dir / f"{storage_name}.owl"

    async def upload_ontology(
        self, ontology_key: str, file: UploadFile, user, description: Optional[str] = None
    ) -> OntologyMetadata:
        ontology_key = self._validate_ontology_key(ontology_key)
        if not file.filename:
            raise ValueError("File must have a filename")
        filename = Path(file.filename).name
        if not filename.lower().endswith(".owl"):
            raise ValueError("File must be in .owl format")

        user_dir = self._get_user_dir(str(user.id))
        metadata = self._load_metadata(user_dir)

        if ontology_key in metadata:
            raise DuplicateOntologyKeyError(f"Ontology key '{ontology_key}' already exists")

        content = await file.read()

        file_path = self._get_ontology_path(user_dir, ontology_key)
        with open(file_path, "wb") as f:
            f.write(content)

        ontology_metadata = {
            "filename": filename,
            "size_bytes": len(content),
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "description": description,
        }
        metadata[ontology_key] = ontology_metadata
        self._save_metadata(user_dir, metadata)

        return OntologyMetadata(
            ontology_key=ontology_key,
            filename=filename,
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
            key = self._validate_ontology_key(key)
            if key not in metadata:
                raise ValueError(f"Ontology key '{key}' not found")

            file_path = self._get_ontology_path(user_dir, key)
            if not file_path.exists():
                raise ValueError(f"Ontology file for key '{key}' not found")

            with open(file_path, "r", encoding="utf-8") as f:
                contents.append(f.read())
        return contents

    def delete_ontology(self, ontology_key: str, user) -> None:
        ontology_key = self._validate_ontology_key(ontology_key)
        user_dir = self._get_user_dir(str(user.id))
        metadata = self._load_metadata(user_dir)

        if ontology_key not in metadata:
            raise ValueError(f"Ontology key '{ontology_key}' not found")

        file_path = self._get_ontology_path(user_dir, ontology_key)

        if file_path.is_file():
            file_path.unlink()

        del metadata[ontology_key]
        self._save_metadata(user_dir, metadata)

    def list_ontologies(self, user) -> dict:
        user_dir = self._get_user_dir(str(user.id))
        return self._load_metadata(user_dir)
