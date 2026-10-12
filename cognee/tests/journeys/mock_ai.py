"""Deterministic stand-ins for the two AI calls cognee makes: LLM and embeddings.

Standalone on purpose: this file has no imports from the test package, so the
quickstart journey can copy it into a fresh virtualenv next to the installed
wheel and get the same behaviour the in-repo journeys use.

What is replaced
----------------
* ``LLMGateway.acreate_structured_output`` is swapped for a dispatcher keyed on
  ``response_model``:

  - ``KnowledgeGraph``: replays a pre-authored graph when the chunk's
    ``Title:`` line names a known corpus document, otherwise falls back to a
    heuristic extractor that turns capitalised phrases into entities. Either
    way the graph is built from the text it was given, so retrieval over it is
    meaningful. Matching is on the title *line*, not on any mention of the
    title: corpus documents cross-reference each other, and a mention must not
    replay another document's graph.
  - ``SummarizedContent``: the pre-authored summary for a corpus document,
    else the first sentence of the chunk.
  - ``str`` (answer completions): echoes the prompt with the question removed.
    A must-contain fact can therefore only pass if retrieval actually surfaced
    it, whichever completion template the retriever used.
  - any other Pydantic model: a structurally valid default instance.

* Embeddings become a hashed bag-of-words vector (md5-bucketed, signed,
  L2-normalised). Two texts that share words land close together under cosine
  distance, so vector search ranks the right chunk first without a network
  call. The real engine class is still constructed, so chunk boundaries and
  vector sizes come from the real configuration.

Failure injection
-----------------
``inject_llm_failure`` wraps whatever ``acreate_structured_output`` is currently
installed (mock or real) and raises for a chosen response model until cleared,
so the interrupted-run journey works in both modes.
"""

from __future__ import annotations

import hashlib
import importlib
import itertools
import math
import os
import re
import typing
from collections.abc import Callable
from typing import Any

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_CAPITALISED_PHRASE_RE = re.compile(
    r"\b(?:[A-Z][a-zA-Z0-9'-]+)(?:\s+(?:[A-Z][a-zA-Z0-9'-]+|of|the|and|&))*"
)
_TITLE_LINE_RE = re.compile(r"^\s*Title:\s*(?P<title>.+?)\s*$", re.MULTILINE)
# Every completion template renders the question as "The question is: `...`".
_QUESTION_RE = re.compile(r"The question is:\s*`.*?`", re.DOTALL | re.IGNORECASE)
_STOPWORD_TEXT = """
    a an the and or but if then of in on at to for from by with about as into like through
    after over between out against during without before under around among is are was were
    be been being have has had do does did will would shall should may might must can could
    it its this that these those he she they them his her their there here what which who
    whom whose why how when where i you we me us my our your not no nor so than too very
"""
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())

_EMBEDDING_ENGINES = (
    (
        "cognee.infrastructure.databases.vector.embeddings.LiteLLMEmbeddingEngine",
        "LiteLLMEmbeddingEngine",
    ),
    (
        "cognee.infrastructure.databases.vector.embeddings.OpenAICompatibleEmbeddingEngine",
        "OpenAICompatibleEmbeddingEngine",
    ),
    (
        "cognee.infrastructure.databases.vector.embeddings.FastembedEmbeddingEngine",
        "FastembedEmbeddingEngine",
    ),
    (
        "cognee.infrastructure.databases.vector.embeddings.OllamaEmbeddingEngine",
        "OllamaEmbeddingEngine",
    ),
)

_ENGINE_CACHES = (
    (
        "cognee.infrastructure.databases.vector.embeddings.get_embedding_engine",
        "create_embedding_engine",
    ),
    ("cognee.infrastructure.databases.vector.create_vector_engine", "_create_vector_engine"),
)

# What ``install_*`` replaced, so ``uninstall_all`` can put it back.
_ORIGINALS: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def tokenize(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS and len(w) > 1]


def hashed_embedding(text: str, dims: int) -> list[float]:
    """Signed hashed bag-of-words, L2-normalised. Deterministic across processes."""
    vector = [0.0] * dims
    tokens = tokenize(text)
    if not tokens:
        # Never return the zero vector: cosine distance is undefined for it.
        vector[0] = 1.0
        return vector
    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dims
        sign = 1.0 if digest[4] & 1 else -1.0
        vector[bucket] += sign
    norm = math.sqrt(sum(v * v for v in vector))
    return [v / norm for v in vector]


async def _mock_embed_text(self, text: list[str]) -> list[list[float]]:
    dims = int(self.get_vector_size())
    return [hashed_embedding(t, dims) for t in text]


def install_mock_embeddings() -> None:
    """Patch every built-in embedding engine so ``embed_text`` hashes locally."""
    for module_name, class_name in _EMBEDDING_ENGINES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:  # optional engines may have missing extras
            continue
        engine_class = getattr(module, class_name, None)
        if engine_class is None:
            continue
        _ORIGINALS.setdefault(f"embed:{class_name}", (engine_class, engine_class.embed_text))
        engine_class.embed_text = _mock_embed_text

    _clear_engine_caches()


def _clear_engine_caches() -> None:
    for module_name, attribute in _ENGINE_CACHES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        target = getattr(module, attribute, None)
        cache_clear = getattr(target, "cache_clear", None)
        if cache_clear is not None:
            cache_clear()


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------


def _first_sentence(text: str, limit: int = 240) -> str:
    body = text
    for marker in ("\n\n", "\n"):
        if marker in body:
            # Skip a leading "Title: ..." line when present.
            parts = [p for p in body.split(marker) if p.strip()]
            if parts and parts[0].lower().startswith("title:") and len(parts) > 1:
                body = marker.join(parts[1:])
            break
    match = re.search(r"(.+?[.!?])(\s|$)", body.strip(), re.DOTALL)
    sentence = match.group(1) if match else body.strip()
    return sentence[:limit].strip() or "Summary unavailable."


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "node"


def heuristic_knowledge_graph(text: str, max_nodes: int = 12):
    """Turn capitalised phrases into entities linked in reading order.

    Used for text that has no pre-authored graph (session Q&A bridged into the
    graph, edited documents, ad-hoc quickstart text). Deterministic.
    """
    from cognee.shared.data_models import Edge, KnowledgeGraph, Node

    seen: dict[str, str] = {}
    for match in _CAPITALISED_PHRASE_RE.finditer(text):
        phrase = match.group(0).strip()
        # Drop sentence-initial single common words like "The".
        if phrase.lower() in _STOPWORDS or len(phrase) < 3:
            continue
        if phrase.startswith("Title:"):
            continue
        key = _slug(phrase)
        if key not in seen:
            seen[key] = phrase
        if len(seen) >= max_nodes:
            break

    nodes = [
        Node(id=key, name=name, type="Entity", description=f"{name}, mentioned in the text.")
        for key, name in seen.items()
    ]
    edges = [
        Edge(
            source_node_id=source,
            target_node_id=target,
            relationship_name="mentioned_with",
            description=None,
        )
        for source, target in itertools.pairwise(seen)
    ]
    return KnowledgeGraph(nodes=nodes, edges=edges)


def _default_instance(model: Any) -> Any:
    """Build a structurally valid instance of an arbitrary Pydantic model."""
    from pydantic import BaseModel, ValidationError

    if model is str:
        return ""
    if model in (int, float, bool):
        return model()
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        return None
    try:
        return model()
    except ValidationError:
        pass
    values = {
        name: _default_for_annotation(field.annotation)
        for name, field in model.model_fields.items()
        if field.is_required()
    }
    try:
        return model(**values)
    except ValidationError:
        return model.model_construct(**values)


def _default_for_annotation(annotation: Any) -> Any:
    from pydantic import BaseModel

    origin = typing.get_origin(annotation)
    args = typing.get_args(annotation)
    if origin in (list, set, tuple):
        return []
    if origin is dict:
        return {}
    if origin is typing.Union or str(origin) == "types.UnionType":
        non_none = [a for a in args if a is not type(None)]
        return _default_for_annotation(non_none[0]) if non_none else None
    if origin is typing.Literal:
        return args[0]
    if annotation is str:
        return ""
    if annotation is int:
        return 0
    if annotation is float:
        return 0.0
    if annotation is bool:
        return False
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _default_instance(annotation)
    return None


def title_of(text: str) -> str | None:
    """The document title named on a ``Title:`` line, if the text has one."""
    match = _TITLE_LINE_RE.search(text)
    return match.group("title") if match else None


def _context_section(prompt: str) -> str:
    """Return the completion prompt with the question removed.

    Every answer template renders the question as ``The question is: `q```
    and the retrieved context elsewhere. Echoing the question would let a
    must-contain fact pass merely because the question mentioned it, so the
    question is cut out and everything else (the context, whichever template
    framed it) is returned.
    """
    return _QUESTION_RE.sub("", prompt).strip()


class MockLLM:
    """Deterministic ``acreate_structured_output`` replacement.

    ``graphs`` maps a corpus title to ``{"knowledge_graph": {...}, "summary": {...}}``.
    A chunk whose ``Title:`` line is that title replays the entry.
    """

    def __init__(self, graphs: dict[str, dict] | None = None):
        self.graphs = graphs or {}
        self.calls: list[tuple[str, str]] = []  # (response_model name, text head)

    def _match(self, text_input: str) -> dict | None:
        title = title_of(text_input)
        return self.graphs.get(title) if title else None

    async def __call__(
        self, text_input: str, system_prompt: str, response_model: Any, **kwargs: Any
    ) -> Any:
        from cognee.shared.data_models import KnowledgeGraph, SummarizedContent

        model_name = getattr(response_model, "__name__", str(response_model))
        self.calls.append((model_name, text_input[:80]))

        if response_model is str:
            return _context_section(text_input)

        if isinstance(response_model, type) and issubclass(response_model, KnowledgeGraph):
            entry = self._match(text_input)
            if entry and entry.get("knowledge_graph"):
                return response_model(**entry["knowledge_graph"])
            return heuristic_knowledge_graph(text_input)

        if isinstance(response_model, type) and issubclass(response_model, SummarizedContent):
            entry = self._match(text_input)
            if entry and entry.get("summary"):
                return response_model(**entry["summary"])
            return response_model(summary=_first_sentence(text_input), description="")

        return _default_instance(response_model)


def install_mock_llm(graphs: dict[str, dict] | None = None) -> MockLLM:
    from cognee.infrastructure.llm.LLMGateway import LLMGateway

    mock = MockLLM(graphs)

    async def _acreate(text_input, system_prompt, response_model, **kwargs):
        return await mock(text_input, system_prompt, response_model, **kwargs)

    _ORIGINALS.setdefault("llm", LLMGateway.__dict__["acreate_structured_output"])
    LLMGateway.acreate_structured_output = staticmethod(_acreate)
    return mock


def install_all(graphs: dict[str, dict] | None = None) -> MockLLM:
    """Install the LLM and embedding mocks and make config happy without keys."""
    os.environ.setdefault("LLM_API_KEY", "mock-key")
    os.environ.setdefault("LLM_PROVIDER", "openai")
    os.environ.setdefault("LLM_MODEL", "openai/gpt-5-mini")
    os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
    os.environ.setdefault("EMBEDDING_MODEL", "openai/text-embedding-3-small")
    os.environ.setdefault("EMBEDDING_DIMENSIONS", "256")
    os.environ.setdefault("EMBEDDING_API_KEY", "mock-key")
    os.environ.setdefault("COGNEE_SKIP_PREFLIGHT", "1")
    os.environ.setdefault("COGNEE_SKIP_CONNECTION_TEST", "true")
    install_mock_embeddings()
    return install_mock_llm(graphs)


def uninstall_all() -> None:
    """Restore the real gateway and embedding engines patched by ``install_all``."""
    from cognee.infrastructure.llm.LLMGateway import LLMGateway

    original_llm = _ORIGINALS.pop("llm", None)
    if original_llm is not None:
        LLMGateway.acreate_structured_output = original_llm
    for key in [k for k in _ORIGINALS if k.startswith("embed:")]:
        engine_class, original = _ORIGINALS.pop(key)
        engine_class.embed_text = original
    _clear_engine_caches()


# ---------------------------------------------------------------------------
# Failure injection (works over the mock or the real gateway)
# ---------------------------------------------------------------------------


class LLMFailureInjector:
    """Raise ``error`` from ``acreate_structured_output`` while armed.

    ``predicate`` receives the response model; default targets KnowledgeGraph
    extraction so cognify fails mid-pipeline, after ingestion succeeded.
    """

    def __init__(self, error: Exception, predicate: Callable[[Any], bool] | None = None):
        self.error = error
        self.predicate = predicate
        self.armed = False
        self.trips = 0
        self._original = None

    def _default_predicate(self, response_model: Any) -> bool:
        from cognee.shared.data_models import KnowledgeGraph

        return isinstance(response_model, type) and issubclass(response_model, KnowledgeGraph)

    def install(self) -> LLMFailureInjector:
        from cognee.infrastructure.llm.LLMGateway import LLMGateway

        original = LLMGateway.acreate_structured_output
        self._original = original
        injector = self

        def _wrapped(text_input, system_prompt, response_model, **kwargs):
            predicate = injector.predicate or injector._default_predicate
            if injector.armed and predicate(response_model):
                injector.trips += 1

                async def _raise():
                    raise injector.error

                return _raise()
            return original(text_input, system_prompt, response_model, **kwargs)

        LLMGateway.acreate_structured_output = staticmethod(_wrapped)
        return self

    def uninstall(self) -> None:
        from cognee.infrastructure.llm.LLMGateway import LLMGateway

        if self._original is not None:
            LLMGateway.acreate_structured_output = (
                self._original
                if isinstance(self._original, staticmethod)
                else staticmethod(self._original)
            )
            self._original = None


def inject_llm_failure(
    error: Exception, predicate: Callable[[Any], bool] | None = None
) -> LLMFailureInjector:
    return LLMFailureInjector(error, predicate).install()
