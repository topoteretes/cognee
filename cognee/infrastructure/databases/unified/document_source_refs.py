"""Which graph artifacts one document owns, read from in-graph provenance."""

from uuid import UUID

from cognee.infrastructure.databases.provenance import EdgeIdentity, parse_source_ref_key


def _document_refs(refs: list[str], data_id: str) -> list[str]:
    """The refs among ``refs`` that name ``data_id``, document- or chunk-scoped."""
    selected = []
    for ref in refs:
        try:
            parsed = parse_source_ref_key(ref)
        except ValueError:
            continue
        if str(parsed.data_id) == data_id:
            selected.append(ref)
    return selected


async def find_document_source_refs(
    graph, dataset_id: UUID | str, data_id: UUID | str
) -> tuple[dict[str, list[str]], dict[EdgeIdentity, list[str]]]:
    """Every ref a document owns in the dataset — v1 doc-scope AND v2 chunk-scope.

    Chunk-scoped ownership (source_ref:v2) means a document's artifacts may
    carry only their producing chunk's ref, so the dataset's ref maps are
    filtered to refs whose data id is this document, whatever their version.
    Returns ``(refs_by_node, refs_by_edge)``; only artifacts the document owns
    appear. Read-only.
    """
    data_id = str(data_id)
    refs_by_node = {
        node_id: document_refs
        for node_id, refs in (await graph.find_node_source_refs_by_dataset(str(dataset_id))).items()
        if (document_refs := _document_refs(refs, data_id))
    }
    refs_by_edge = {
        edge: document_refs
        for edge, refs in (await graph.find_edge_source_refs_by_dataset(str(dataset_id))).items()
        if (document_refs := _document_refs(refs, data_id))
    }
    return refs_by_node, refs_by_edge
