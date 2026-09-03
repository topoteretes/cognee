import os
from pathlib import Path
from typing import Any

from cognee.infrastructure.files.storage import get_file_storage, get_storage_config
from cognee.infrastructure.files.utils.get_file_metadata import get_file_metadata
from cognee.infrastructure.loaders.LoaderInterface import LoaderInterface, LoaderResult
from cognee.infrastructure.loaders.store_derived_text import store_derived_text

# Source-file extensions of the languages enola extracts (its per-language
# extractors and tree-sitter grammars; see https://github.com/enola-labs/enola
# "Supported Languages"), each verified against enola v0.4.12 to produce facts
# for a single staged file (js/jsx need a tsconfig.json detection marker — see
# cognee.tasks.code_graph.code_files._DETECTION_MARKERS; a lone .kts or C .h
# yields no facts on its own but is still code). Deliberately excludes
# extensions that double as generic config/markup (yml/yaml for Ansible, json
# for OpenAPI, xaml/razor/cshtml/hbs templates, md — enola now reads markdown
# pages too, but they are documents first) — claiming those would hijack
# ordinary documents — and graphql, which enola only reads inside a larger
# project context.
SUPPORTED_CODE_EXTENSIONS = frozenset(
    {
        "c",
        "cc",
        "cpp",
        "cs",
        "cxx",
        "dart",
        "fs",
        "go",
        "h",
        "hcl",
        "hh",
        "hpp",
        "java",
        "js",
        "jsx",
        "kt",
        "kts",
        "php",
        "proto",
        "py",
        "rake",
        "rb",
        "rs",
        "scala",
        "svelte",
        "swift",
        "tf",
        "ts",
        "tsx",
        "vb",
        "vue",
    }
)


class CodeLoader(LoaderInterface):
    """Loader for source-code files the enola code graph pipeline can extract.

    Selecting this loader IS the add-time recognition of code files: it
    returns a ``LoaderResult`` carrying the ``system_metadata`` route stamp
    ({"source": "code"}), which ingest_data writes onto the Data record and
    cognify routes down the CODE route — the same mechanism the DLT CSV
    loader uses. Content is stored verbatim (code is text), but under its
    real extension — content sniffing reports code as plain text, so the
    storage file name is the only place the extension survives for enola's
    language detection.

    Must be matched by file-name extension only: content-based detection can
    never identify a code language, and the engine's content fallback reports
    "txt", which this loader deliberately does not claim.
    """

    loader_name = "code_loader"

    @property
    def supported_extensions(self) -> list[str]:
        return sorted(SUPPORTED_CODE_EXTENSIONS)

    @property
    def supported_mime_types(self) -> list[str]:
        return ["text/x-script", "text/x-source"]

    def can_handle(self, extension: str, mime_type: str) -> bool:
        return (extension or "").lower() in SUPPORTED_CODE_EXTENSIONS

    async def load(
        self, file_path: str, encoding: str = "utf-8", **kwargs: Any
    ) -> "str | LoaderResult":
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, "rb") as f:
            file_metadata = await get_file_metadata(f)
        # Content-hash name like the text loader, but keeping the code extension.
        storage_file_name = "code_" + file_metadata["content_hash"] + Path(file_path).suffix.lower()

        with open(file_path, encoding=encoding) as f:
            content = f.read()

        if not kwargs.get("persist", True):
            return content

        storage_config = get_storage_config()
        data_root_directory = storage_config["data_root_directory"]
        storage = get_file_storage(data_root_directory)

        return await store_derived_text(
            storage, storage_file_name, content, system_metadata={"source": "code"}
        )
