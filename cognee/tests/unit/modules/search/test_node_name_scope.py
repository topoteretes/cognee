"""node_name means a node-set tag or a graph scope, set by node_type; each search type honours some."""

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
        (None, None, None),
        (NodeSet, ["tag"], NodeNameScope.NODE_SET),
        (TextDocument, ["doc_a"], NodeNameScope.GRAPH),
        (PdfDocument, ["doc_a"], NodeNameScope.GRAPH),
        (Entity, ["Apple"], NodeNameScope.GRAPH),
    ],
)
def test_scope_follows_node_type(node_type, node_name, expected):
    assert node_name_scope(node_type, node_name) is expected


def test_names_without_a_node_type_are_refused():
    with pytest.raises(QueryValidationError, match="node_name needs a node_type"):
        node_name_scope(None, ["Apple"])


def test_types_that_ignore_node_name_are_not_in_the_table():
    for search_type in UNSUPPORTING:
        assert search_type not in SUPPORTED_SCOPES


def test_chunk_types_honour_every_scope():
    for search_type in (SearchType.CHUNKS, SearchType.RAG_COMPLETION):
        assert SUPPORTED_SCOPES[search_type] == frozenset(NodeNameScope)
    assert SearchType.CHUNKS in search_types_honouring(NodeNameScope.GRAPH)
    assert SearchType.TEMPORAL not in search_types_honouring(NodeNameScope.GRAPH)


@pytest.mark.parametrize("search_type", UNSUPPORTING)
def test_unsupporting_type_with_node_name_is_refused(search_type):
    with pytest.raises(QueryValidationError, match="does not apply node_name"):
        reject_unsupported_node_name(search_type, NodeSet, ["tag"])


@pytest.mark.parametrize("search_type", UNSUPPORTING)
def test_unsupporting_type_without_node_name_is_fine(search_type):
    assert reject_unsupported_node_name(search_type, NodeSet, None) is None


@pytest.mark.parametrize("search_type", [SearchType.CHUNKS, SearchType.RAG_COMPLETION])
@pytest.mark.parametrize("node_type", [TextDocument, Entity])
def test_chunk_types_accept_a_graph_scope(search_type, node_type):
    assert reject_unsupported_node_name(search_type, node_type, ["x"]) is NodeNameScope.GRAPH


def test_temporal_refuses_a_graph_scope():
    with pytest.raises(QueryValidationError, match="TEMPORAL does not apply"):
        reject_unsupported_node_name(SearchType.TEMPORAL, TextDocument, ["doc_a"])


def test_refusal_names_the_types_that_honour_the_scope():
    with pytest.raises(QueryValidationError, match="CHUNKS, GRAPH_COMPLETION"):
        reject_unsupported_node_name(SearchType.SUMMARIES, TextDocument, ["doc_a"])


def test_lucky_types_are_unbounded_without_a_filter():
    assert lucky_search_types(NodeSet, None) is None


def test_lucky_types_with_a_graph_scope_exclude_types_that_drop_it():
    allowed = lucky_search_types(TextDocument, ["doc_a"])
    assert allowed is not None
    assert {SearchType.CHUNKS, SearchType.RAG_COMPLETION, SearchType.GRAPH_COMPLETION} <= allowed
    assert SearchType.HYBRID_COMPLETION in allowed  # defers to GRAPH_COMPLETION
    assert allowed.isdisjoint(UNSUPPORTING)
    assert SearchType.TEMPORAL not in allowed
