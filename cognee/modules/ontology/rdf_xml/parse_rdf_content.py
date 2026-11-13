"""Parse RDF content and files without letting the parser reach outside them.

Uploaded ontologies are stored as ``.owl`` whatever their real syntax, so they are
parsed by trying rdflib formats in turn. rdflib's JSON-LD parser fetches every
``@context`` / ``@import`` given as a string (a URL, a relative path or a ``file:``
path), so a crafted document would make the server send requests or read local
files whenever it is parsed. rdflib has no switch for this, so such documents are
refused before the JSON-LD parser sees them; contexts written out inline still parse.
"""

import json
from collections.abc import Iterable

from rdflib import Graph
from rdflib.parser import Parser
from rdflib.plugin import PluginException
from rdflib.plugin import get as get_plugin
from rdflib.plugins.parsers.jsonld import JsonLDParser
from rdflib.util import guess_format

from cognee.shared.logging_utils import get_logger

logger = get_logger(__name__)

# Uploads are stored as ``.owl`` whatever their real syntax: RDF/XML first, then the rest.
FALLBACK_FORMATS = ("xml", "turtle", "n3", "nt", "json-ld", "trig", "nquads")

_CONTEXT_KEYS = ("@context", "@import")


def _is_jsonld_format(rdf_format: str) -> bool:
    """True for any name rdflib resolves to its JSON-LD parser (``json-ld``, mime aliases)."""
    try:
        return issubclass(get_plugin(rdf_format, Parser), JsonLDParser)
    except PluginException:
        return False


def _remote_context_reference(node) -> str | None:
    """The first ``@context`` / ``@import`` value that names a document instead of inlining it."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key in _CONTEXT_KEYS:
                references = value if isinstance(value, list) else [value]
                for reference in references:
                    if isinstance(reference, str):
                        return reference
            found = _remote_context_reference(value)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _remote_context_reference(item)
            if found is not None:
                return found
    return None


def refuse_remote_jsonld_contexts(content: str | bytes) -> None:
    """Raise ``ValueError`` unless ``content`` is JSON whose contexts are all inline.

    Content that is not JSON is refused too: the JSON-LD parser could not read it
    either, and this check must never let through a document it did not inspect.
    """
    document = json.loads(content)
    reference = _remote_context_reference(document)
    if reference is not None:
        raise ValueError(
            f"JSON-LD context {reference!r} references an external document; "
            "only inline contexts are supported"
        )


def parse_rdf_content(content: str | bytes, formats: Iterable[str]) -> tuple[Graph, str]:
    """Parse ``content`` with the first of ``formats`` rdflib accepts.

    Returns the graph and the format that parsed it. Raises ``ValueError`` naming
    each format's error when none does.
    """
    parse_errors = []
    for rdf_format in formats:
        graph = Graph()
        try:
            if _is_jsonld_format(rdf_format):
                refuse_remote_jsonld_contexts(content)
            graph.parse(data=content, format=rdf_format)
        except Exception as error:
            # Each rdflib parser raises its own exception type; a failed guess is expected.
            logger.debug("RDF content is not %s, trying next format", rdf_format, exc_info=True)
            parse_errors.append(f"{rdf_format}: {error}")
            continue
        return graph, rdf_format

    raise ValueError("Tried formats: " + "; ".join(parse_errors))


def parse_rdf_file(graph: Graph, file_path: str, rdf_format: str | None = None) -> None:
    """Parse the local file ``file_path`` into ``graph``.

    Without ``rdf_format`` rdflib picks the format from the extension (turtle when
    it cannot tell). The base IRI is the file's location, as with ``Graph.parse``;
    a file read as JSON-LD is checked by ``refuse_remote_jsonld_contexts`` first.
    """
    rdf_format = rdf_format or guess_format(file_path)
    if rdf_format is not None and _is_jsonld_format(rdf_format):
        with open(file_path, "rb") as file:
            refuse_remote_jsonld_contexts(file.read())
    graph.parse(file_path, format=rdf_format)
