import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from fastapi import UploadFile

from cognee.base_config import get_base_config
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import (
    detect_rdf_format,
    normalize_content_type,
    supported_rdf_content_types,
    supported_rdf_extensions,
)

_ONTOLOGY_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class DuplicateOntologyKeyError(ValueError):
    """Raised when an ontology key already exists for the user.

    A dedicated type lets the router surface a safe, literal client message
    without echoing the exception text (avoids stack-trace exposure).
    """


class UnsupportedOntologyFormatError(ValueError):
    """Raised when neither the filename extension nor the content type names an RDF format.

    The message is built only from the static accepted-format lists (never from
    the upload itself), so the router can return it to the client verbatim.
    """

    def __init__(self) -> None:
        super().__init__(
            "Unsupported ontology format. Accepted file extensions: "
            f"{', '.join(supported_rdf_extensions())}. "
            "Accepted content types (for filenames without a known extension): "
            f"{', '.join(supported_rdf_content_types())}."
        )


class OntologyFile(BytesIO):
    """Stored ontology bytes plus the upload's filename and content type.

    ``RDFLibOntologyResolver`` reads ``filename``/``name`` and ``content_type``
    off the file object to pick the parser, so both must survive the round
    trip through storage.
    """

    def __init__(self, content: bytes, filename: str, content_type: str | None) -> None:
        super().__init__(content)
        self.filename = filename
        self.name = filename
        self.content_type = content_type


@dataclass
class OntologyMetadata:
    ontology_key: str
    filename: str
    size_bytes: int
    uploaded_at: str
    description: str | None = None


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
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_metadata(self, user_dir: Path, metadata: dict):
        metadata_path = self._get_metadata_path(user_dir)
        with open(metadata_path, "w", encoding="utf-8") as f:
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
        self, ontology_key: str, file: UploadFile, user, description: str | None = None
    ) -> OntologyMetadata:
        ontology_key = self._validate_ontology_key(ontology_key)
        if not file.filename:
            raise ValueError("File must have a filename")
        filename = Path(file.filename).name
        content_type = normalize_content_type(file.content_type)
        rdf_format = detect_rdf_format(filename, content_type)
        if rdf_format is None:
            raise UnsupportedOntologyFormatError()

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
            "content_type": content_type,
            "format": rdf_format,
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
        ontology_key: list[str],
        files: list[UploadFile],
        user,
        descriptions: list[str] | None = None,
    ) -> list[OntologyMetadata]:
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

    def _locate_ontologies(self, ontology_key: list[str], user) -> list[tuple[Path, dict]]:
        """Resolve each key to its stored file path and metadata entry.

        Raises:
            ValueError: If any ontology key (or its file) is not found
        """
        user_dir = self._get_user_dir(str(user.id))
        metadata = self._load_metadata(user_dir)

        located = []
        for key in ontology_key:
            key = self._validate_ontology_key(key)
            if key not in metadata:
                raise ValueError(f"Ontology key '{key}' not found")

            file_path = self._get_ontology_path(user_dir, key)
            if not file_path.exists():
                raise ValueError(f"Ontology file for key '{key}' not found")

            located.append((file_path, metadata[key]))
        return located

    def get_ontology_contents(self, ontology_key: list[str], user) -> list[str]:
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
        contents = []
        for file_path, _ in self._locate_ontologies(ontology_key, user):
            with open(file_path, "r", encoding="utf-8") as f:
                contents.append(f.read())
        return contents

    def get_ontology_files(self, ontology_key: list[str], user) -> list[OntologyFile]:
        """
        Retrieve ontologies as file objects ready for ``RDFLibOntologyResolver``.

        Unlike :meth:`get_ontology_contents`, each returned object carries the
        original upload filename and content type so the resolver can pick the
        right RDF parser (Turtle, JSON-LD, ...) instead of trial-parsing.

        Raises:
            ValueError: If any ontology key not found
        """
        files = []
        for file_path, entry in self._locate_ontologies(ontology_key, user):
            with open(file_path, "rb") as f:
                content = f.read()
            files.append(
                OntologyFile(
                    content,
                    filename=entry.get("filename") or file_path.name,
                    content_type=entry.get("content_type"),
                )
            )
        return files

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
