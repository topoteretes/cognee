import asyncio
import hashlib
import json
import os
import posixpath
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from fastapi import UploadFile
from rdflib import Graph

from cognee.base_config import get_base_config
from cognee.infrastructure.files.storage import get_file_storage
from cognee.infrastructure.files.storage.StorageManager import StorageManager
from cognee.modules.ontology.rdf_xml.parse_rdf_content import FALLBACK_FORMATS, parse_rdf_content
from cognee.modules.ontology.rdf_xml.RDFLibOntologyResolver import (
    detect_rdf_format,
    normalize_content_type,
    supported_rdf_content_types,
    supported_rdf_extensions,
)

_ONTOLOGY_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

# Query-parameter name -> (rdflib serializer name, media type, file extension).
EXPORT_FORMATS: dict[str, tuple[str, str, str]] = {
    "turtle": ("turtle", "text/turtle", "ttl"),
    "xml": ("xml", "application/rdf+xml", "rdf"),
    "json-ld": ("json-ld", "application/ld+json", "jsonld"),
    "nt": ("nt", "application/n-triples", "nt"),
}


class DuplicateOntologyKeyError(ValueError):
    """Raised when an ontology key already exists for the user.

    A dedicated type lets the router surface a safe, literal client message
    without echoing the exception text (avoids stack-trace exposure).
    """


def unsupported_ontology_format_message() -> str:
    """Client-facing text for a rejected upload format.

    Built only from the static accepted-format lists (never from the upload
    itself), so the router can send it verbatim without echoing request data.
    """
    return (
        "Unsupported ontology format. Accepted file extensions: "
        f"{', '.join(supported_rdf_extensions())}. "
        "Accepted content types (for filenames without a known extension): "
        f"{', '.join(supported_rdf_content_types())}."
    )


class UnsupportedOntologyFormatError(ValueError):
    """Raised when neither the filename extension nor the content type names an RDF format.

    Carries :func:`unsupported_ontology_format_message`; the router calls that
    function itself rather than stringifying the exception.
    """

    def __init__(self) -> None:
        super().__init__(unsupported_ontology_format_message())


class OntologyFile(BytesIO):
    """Stored ontology bytes plus the upload's filename, content type and format.

    ``RDFLibOntologyResolver`` reads ``rdf_format`` (the parser name recorded at
    upload time), then ``filename``/``name`` and ``content_type``, off the file
    object to pick the parser, so all of them must survive the round trip
    through storage.
    """

    def __init__(
        self,
        content: bytes,
        filename: str,
        content_type: str | None,
        rdf_format: str | None = None,
    ) -> None:
        super().__init__(content)
        self.filename = filename
        self.name = filename
        self.content_type = content_type
        self.rdf_format = rdf_format


class OntologyNotFoundError(ValueError):
    """Raised when an ontology key is invalid or not uploaded by this user.

    A key owned by another user is indistinguishable from a missing one.
    """


class OntologyParseError(Exception):
    """Raised when a stored ontology file cannot be parsed as RDF by rdflib.

    Deliberately not a ``ValueError`` so the router can tell "key not found"
    (404) apart from "file exists but is not valid RDF" (422).
    """


class OntologySerializationError(Exception):
    """Raised when a parsed ontology cannot be written in the requested format.

    rdflib's RDF/XML writer, for one, refuses predicates it cannot shorten to a
    QName (an IRI ending in digits, such as ``http://example.org/123``).
    """


@dataclass
class OntologyMetadata:
    ontology_key: str
    filename: str
    size_bytes: int
    uploaded_at: str
    description: str | None = None


class OntologyService:
    """Per-user store of uploaded ontologies, kept in cognee's file storage.

    Layout under the data root: ``<user id>/metadata.json`` (the index, keyed by
    ontology key) and ``<user id>/<blake2s of the key>.owl``. Every read and write
    goes through ``get_file_storage``, so the files follow the data root (to S3
    when it is an ``s3://`` URL) and are sealed when file encryption is on.

    The index is read, changed and written back without a lock: two concurrent
    uploads or deletes by one user can lose an entry.
    """

    def __init__(self):
        pass

    @property
    def base_dir(self) -> Path:
        base_config = get_base_config()
        return base_config.data_root_directory

    def _storage(self) -> StorageManager:
        return get_file_storage(os.fspath(self.base_dir))

    def _get_user_dir(self, user_id: str) -> str:
        """The user's folder as a path relative to the data root.

        Raises:
            ValueError: If the user id would leave the data root
        """
        # Checked against a stand-in root, so the rule is the same for a local
        # folder and an S3 prefix.
        root = "/data-root"
        user_dir = posixpath.normpath(posixpath.join(root, str(user_id)))
        if user_dir != root and not user_dir.startswith(f"{root}/"):
            raise ValueError("Invalid user id")
        relative_dir = posixpath.relpath(user_dir, root)
        return "" if relative_dir == "." else relative_dir

    def _get_metadata_path(self, user_dir: str) -> str:
        return posixpath.join(user_dir, "metadata.json")

    async def _load_metadata(self, user_dir: str) -> dict:
        metadata_path = self._get_metadata_path(user_dir)
        if await self._storage().file_exists(metadata_path):
            return json.loads(await self._read_file(metadata_path))
        return {}

    async def _save_metadata(self, user_dir: str, metadata: dict):
        await self._storage().store(
            self._get_metadata_path(user_dir), json.dumps(metadata, indent=2), overwrite=True
        )

    async def _read_file(self, file_path: str) -> bytes:
        async with self._storage().open(file_path, "rb") as file:
            # The read is blocking IO (a network request on S3): keep it off the event loop.
            return await asyncio.to_thread(file.read)

    def _validate_ontology_key(self, ontology_key: str) -> str:
        normalized_key = ontology_key.strip()
        if not _ONTOLOGY_KEY_RE.fullmatch(normalized_key):
            raise ValueError("Invalid ontology key")
        return normalized_key

    def _get_ontology_path(self, user_dir: str, ontology_key: str) -> str:
        normalized_key = self._validate_ontology_key(ontology_key)
        storage_name = hashlib.blake2s(normalized_key.encode("utf-8"), digest_size=16).hexdigest()
        return posixpath.join(user_dir, f"{storage_name}.owl")

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
        metadata = await self._load_metadata(user_dir)

        if ontology_key in metadata:
            raise DuplicateOntologyKeyError(f"Ontology key '{ontology_key}' already exists")

        content = await file.read()

        await self._storage().store(
            self._get_ontology_path(user_dir, ontology_key), BytesIO(content), overwrite=True
        )

        ontology_metadata = {
            "filename": filename,
            "content_type": content_type,
            "format": rdf_format,
            "size_bytes": len(content),
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "description": description,
        }
        metadata[ontology_key] = ontology_metadata
        await self._save_metadata(user_dir, metadata)

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

    async def _locate_ontologies(self, ontology_key: list[str], user) -> list[tuple[str, dict]]:
        """Resolve each key to its stored file path and metadata entry.

        Raises:
            ValueError: If any ontology key (or its file) is not found
        """
        user_dir = self._get_user_dir(str(user.id))
        metadata = await self._load_metadata(user_dir)
        storage = self._storage()

        located = []
        for key in ontology_key:
            key = self._validate_ontology_key(key)
            if key not in metadata:
                raise ValueError(f"Ontology key '{key}' not found")

            file_path = self._get_ontology_path(user_dir, key)
            if not await storage.is_file(file_path):
                raise ValueError(f"Ontology file for key '{key}' not found")

            located.append((file_path, metadata[key]))
        return located

    async def get_ontology_files(self, ontology_key: list[str], user) -> list[OntologyFile]:
        """
        Retrieve ontologies as file objects ready for ``RDFLibOntologyResolver``.

        This is the one way a stored ontology is loaded for cognify/remember
        (REST routers and the MCP server alike). Each returned object carries
        the format detected at upload time plus the original filename and
        content type, so the resolver picks the right RDF parser (Turtle,
        JSON-LD, ...) instead of trial-parsing.

        Raises:
            ValueError: If any ontology key not found
        """
        files = []
        for file_path, entry in await self._locate_ontologies(ontology_key, user):
            files.append(
                OntologyFile(
                    await self._read_file(file_path),
                    filename=entry.get("filename") or posixpath.basename(file_path),
                    content_type=entry.get("content_type"),
                    rdf_format=entry.get("format"),
                )
            )
        return files

    async def delete_ontology(self, ontology_key: str, user) -> None:
        ontology_key = self._validate_ontology_key(ontology_key)
        user_dir = self._get_user_dir(str(user.id))
        metadata = await self._load_metadata(user_dir)

        if ontology_key not in metadata:
            raise ValueError(f"Ontology key '{ontology_key}' not found")

        # A file that is already gone is not an error: the index entry still goes.
        await self._storage().remove(self._get_ontology_path(user_dir, ontology_key))

        del metadata[ontology_key]
        await self._save_metadata(user_dir, metadata)

    async def list_ontologies(self, user) -> dict:
        user_dir = self._get_user_dir(str(user.id))
        return await self._load_metadata(user_dir)

    async def load_ontology_graph(self, ontology_key: str, user) -> Graph:
        """
        Parse the stored ontology for ``ontology_key`` into an ``rdflib.Graph``.

        Scoped to the user's own directory and metadata, exactly like
        ``delete_ontology``: a key uploaded by another user is "not found".

        Raises:
            OntologyNotFoundError: If the key is invalid or not found for this user
            OntologyParseError: If the stored file is not parseable RDF
        """
        try:
            ((file_path, entry),) = await self._locate_ontologies([ontology_key], user)
        except ValueError as error:
            raise OntologyNotFoundError(str(error)) from error

        # Uploads are stored as ``.owl`` whatever their real syntax. The format
        # recorded at upload time is tried first; uploads that predate the
        # ``format`` field fall back to probing every syntax.
        candidate_formats = list(FALLBACK_FORMATS)
        stored_format = entry.get("format")
        if stored_format in candidate_formats:
            candidate_formats.remove(stored_format)
            candidate_formats.insert(0, stored_format)

        content = await self._read_file(file_path)
        try:
            # Parsing with rdflib is CPU-bound: keep it off the event loop.
            graph, _ = await asyncio.to_thread(parse_rdf_content, content, candidate_formats)
        except ValueError as error:
            raise OntologyParseError(
                f"Unable to parse ontology '{ontology_key}'. {error}"
            ) from error
        return graph

    async def export_ontology(
        self, ontology_key: str, user, export_format: str = "turtle"
    ) -> bytes:
        """
        Serialize the stored ontology for ``ontology_key`` in ``export_format``.

        ``export_format`` must be one of ``EXPORT_FORMATS`` ("turtle", "xml",
        "json-ld", "nt"). Returns UTF-8 encoded bytes.

        Raises:
            KeyError: If ``export_format`` is not a supported export format
            OntologyNotFoundError: If the key is invalid or not found for this user
            OntologyParseError: If the stored file is not parseable RDF
            OntologySerializationError: If the ontology cannot be written in ``export_format``
        """
        rdflib_format = EXPORT_FORMATS[export_format][0]
        graph = await self.load_ontology_graph(ontology_key, user)
        try:
            # Serializing with rdflib is CPU-bound too.
            serialized = await asyncio.to_thread(
                graph.serialize, format=rdflib_format, encoding="utf-8"
            )
        except ValueError as error:
            raise OntologySerializationError(
                f"Ontology '{ontology_key}' cannot be serialized as {export_format}: {error}"
            ) from error
        if isinstance(serialized, str):
            return serialized.encode("utf-8")
        return serialized
