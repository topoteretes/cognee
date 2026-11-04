"""node_name means one of three things, set by node_type; each search type honours some."""

import pytest

from cognee.modules.data.processing.document_types import PdfDocument, TextDocument
from cognee.modules.engine.models import Entity
from cognee.modules.engine.models.node_set import NodeSet
from cognee.modules.retrieval.exceptions.exceptions import QueryValidationError
from cognee.modules.search.methods.node_name_scope import (
    SUPPORTED_SCOPES,
    NodeNameScope,
    lucky_search_types,
    node_name_scope,
    reject_unsupported_node_name,
    search_types_honouring,
)
from cognee.modules.search.types import SearchType

UNSUPPORTING = [
    SearchType.SUMMARIES,
    SearchType.CHUNKS_LEXICAL,
    SearchType.CYPHER,
    SearchType.NATURAL_LANGUAGE,
    SearchType.GRAPH_REPORT,
    SearchType.CODE,
    SearchType.SKILLS,
    SearchType.BROAD,
]


@pytest.mark.parametrize(
    ("node_type", "node_name", "expected"),
    [
        (NodeSet, None, None),
        (NodeSet, [], None),
        (TextDocument, None, None),
        (NodeSet, ["tag"], NodeNameScope.NODE_SET),
        (TextDocument, ["doc_a"], NodeNameScope.DOCUMENT),
        (PdfDocument, ["doc_a"], NodeNameScope.DOCUMENT),
        (Entity, ["Apple"], NodeNameScope.GRAPH),
        (None, ["Apple"], NodeNameScope.GRAPH),
    ],
)
def test_scope_follows_node_type(node_type, node_name, expected):
    assert node_name_scope(node_type, node_name) is expected


def test_types_that_ignore_node_name_are_not_in_the_table():
    for search_type in UNSUPPORTING:
        assert search_type not in SUPPORTED_SCOPES


def test_chunk_types_honour_node_sets_and_documents_only():
    for search_type in (SearchType.CHUNKS, SearchType.RAG_COMPLETION):
        assert SUPPORTED_SCOPES[search_type] == {NodeNameScope.NODE_SET, NodeNameScope.DOCUMENT}
    assert SearchType.CHUNKS in search_types_honouring(NodeNameScope.DOCUMENT)
    assert SearchType.CHUNKS not in search_types_honouring(NodeNameScope.GRAPH)


@pytest.mark.parametrize("search_type", UNSUPPORTING)
def test_unsupporting_type_with_node_name_is_refused(search_type):
    with pytest.raises(QueryValidationError, match="does not apply node_name"):
        reject_unsupported_node_name(search_type, NodeSet, ["tag"], "OR")


@pytest.mark.parametrize("search_type", UNSUPPORTING)
def test_unsupporting_type_without_node_name_is_fine(search_type):
    assert reject_unsupported_node_name(search_type, NodeSet, None, "OR") is None


@pytest.mark.parametrize("search_type", [SearchType.CHUNKS, SearchType.RAG_COMPLETION])
def test_chunk_types_refuse_a_graph_scope(search_type):
    with pytest.raises(QueryValidationError, match="node_type=Entity"):
        reject_unsupported_node_name(search_type, Entity, ["Apple"], "OR")


def test_temporal_refuses_a_document_scope():
    with pytest.raises(QueryValidationError, match="TEMPORAL does not apply"):
        reject_unsupported_node_name(SearchType.TEMPORAL, TextDocument, ["doc_a"], "OR")


def test_refusal_names_the_types_that_honour_the_scope():
    with pytest.raises(QueryValidationError, match="CHUNKS, GRAPH_COMPLETION"):
        reject_unsupported_node_name(SearchType.SUMMARIES, TextDocument, ["doc_a"], "OR")


def test_document_scope_returns_its_scope():
    assert (
        reject_unsupported_node_name(SearchType.CHUNKS, TextDocument, ["doc_a"], "OR")
        is NodeNameScope.DOCUMENT
    )


def test_and_over_several_documents_is_refused():
    with pytest.raises(QueryValidationError, match="exactly one document"):
        reject_unsupported_node_name(SearchType.CHUNKS, TextDocument, ["doc_a", "doc_b"], "AND")


def test_and_over_one_document_is_allowed():
    scope = reject_unsupported_node_name(SearchType.CHUNKS, TextDocument, ["doc_a", "doc_a"], "AND")
    assert scope is NodeNameScope.DOCUMENT


def test_and_over_several_node_sets_is_allowed():
    scope = reject_unsupported_node_name(SearchType.CHUNKS, NodeSet, ["a", "b"], "AND")
    assert scope is NodeNameScope.NODE_SET


def test_lucky_types_are_unbounded_without_a_filter():
    assert lucky_search_types(NodeSet, None) is None


def test_lucky_types_with_a_document_filter_exclude_types_that_drop_it():
    allowed = lucky_search_types(TextDocument, ["doc_a"])
    assert allowed is not None
    assert {SearchType.CHUNKS, SearchType.RAG_COMPLETION, SearchType.GRAPH_COMPLETION} <= allowed
    assert SearchType.HYBRID_COMPLETION in allowed  # defers to GRAPH_COMPLETION
    assert allowed.isdisjoint(UNSUPPORTING)
    assert SearchType.TEMPORAL not in allowed
