"""Test-only fake GraphVectorStore + tests pinning graph-native delete/rollback semantics.

This module is the *reference semantics* for Part 2 of the graph-native
provenance plan. Part 2 wires ``delete`` / ``rollback`` onto
``GraphVectorStoreInterface``; until a real backend implements the capability,
these fakes let Part 2's tests run with zero real-DB / engine bootstrap.

``FakeGraphVectorStore`` is a dependency-light, in-memory store that holds
nodes and edges, each tagged with the graph-native provenance the contract
defines (``source_refs``, ``source_run_refs``, ``dataset_ids``). It implements
the three deletion entry points with the ref-counting rules the contract
mandates:

* ``delete_by_source_ref``   — strip one ingestion source ref; hard-delete an
  artifact only when its *last* source ref is gone, otherwise *detach* it.
* ``delete_by_dataset_id``   — strip one dataset; hard-delete an artifact only
  when it belonged to no other dataset, otherwise *detach* it.
* ``rollback_by_pipeline_run_id`` — strip one run's source-run ref; hard-delete
  an artifact only when it has no remaining source ref AND no other source-run
  ref (i.e. the run *solely* introduced it), otherwise *detach* it.

Every operation returns a ``ProvenanceDeleteResult`` with accurate
deleted/detached counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4

import pytest

from cognee.infrastructure.databases.unified.graph_vector_store_interface import (
    GraphVectorStoreInterface,
)
from cognee.modules.graph.provenance import (
    EdgeIdentity,
    ProvenanceDeleteResult,
    UnsupportedProvenanceCapability,
    make_source_ref,
    make_source_run_ref,
)


# ---------------------------------------------------------------------------
# In-memory artifact records
# ---------------------------------------------------------------------------
@dataclass
class _FakeArtifact:
    """A single in-memory graph artifact (node or edge) with its provenance.

    Refs are stored as mutable sets so the deletion primitives can strip a ref
    in place and then decide hard-delete vs. detach from what remains. This is
    the in-memory stand-in for the ``source_refs`` / ``source_run_refs`` /
    ``dataset_ids`` lists the contract stamps onto real graph artifacts.
    """

    key: object
    source_refs: set[str] = field(default_factory=set)
    source_run_refs: set[str] = field(default_factory=set)
    dataset_ids: set[UUID] = field(default_factory=set)


class FakeGraphVectorStore(GraphVectorStoreInterface):
    """In-memory ``GraphVectorStoreInterface`` with the contract's ref-counting.

    Holds nodes keyed by ``node_id`` and edges keyed by an ``EdgeIdentity``.
    Each artifact carries the provenance sets the graph-native contract defines.
    The three deletion methods mutate those sets and report exactly what was
    hard-deleted versus detached.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, _FakeArtifact] = {}
        self.edges: dict[EdgeIdentity, _FakeArtifact] = {}

    # -- capability flag ----------------------------------------------------
    def supports_graph_native_delete(self) -> bool:
        return True

    # -- seeding helpers (test-only) ---------------------------------------
    def add_node(
        self,
        node_id: str,
        *,
        source_refs: tuple[str, ...] = (),
        source_run_refs: tuple[str, ...] = (),
        dataset_ids: tuple[UUID, ...] = (),
    ) -> None:
        """Insert a node with its provenance. Re-adding merges the ref sets."""
        artifact = self.nodes.get(node_id)
        if artifact is None:
            artifact = _FakeArtifact(key=node_id)
            self.nodes[node_id] = artifact
        artifact.source_refs.update(source_refs)
        artifact.source_run_refs.update(source_run_refs)
        artifact.dataset_ids.update(dataset_ids)

    def add_edge(
        self,
        identity: EdgeIdentity,
        *,
        source_refs: tuple[str, ...] = (),
        source_run_refs: tuple[str, ...] = (),
        dataset_ids: tuple[UUID, ...] = (),
    ) -> None:
        """Insert an edge with its provenance. Re-adding merges the ref sets."""
        artifact = self.edges.get(identity)
        if artifact is None:
            artifact = _FakeArtifact(key=identity)
            self.edges[identity] = artifact
        artifact.source_refs.update(source_refs)
        artifact.source_run_refs.update(source_run_refs)
        artifact.dataset_ids.update(dataset_ids)

    # -- core ref-counting engine ------------------------------------------
    @staticmethod
    def _apply(
        store: dict,
        *,
        select,
        strip,
        survives,
    ) -> tuple[int, int]:
        """Apply a strip-then-decide pass over one artifact store.

        - ``select(artifact)``: True if the artifact is touched by this op.
        - ``strip(artifact)``: mutate the artifact to remove the targeted ref(s).
        - ``survives(artifact)``: True if the stripped artifact should be
          detached (kept), False if it should be hard-deleted.

        Returns ``(deleted_count, detached_count)``.
        """
        deleted = 0
        detached = 0
        to_remove = []
        for key, artifact in store.items():
            if not select(artifact):
                continue
            strip(artifact)
            if survives(artifact):
                detached += 1
            else:
                deleted += 1
                to_remove.append(key)
        for key in to_remove:
            del store[key]
        return deleted, detached

    async def delete_by_source_ref(self, source_ref: str) -> ProvenanceDeleteResult:
        def select(a: _FakeArtifact) -> bool:
            return source_ref in a.source_refs

        def strip(a: _FakeArtifact) -> None:
            a.source_refs.discard(source_ref)

        # Detach only if another source ref still keeps the artifact alive.
        def survives(a: _FakeArtifact) -> bool:
            return bool(a.source_refs)

        n_del, n_det = self._apply(self.nodes, select=select, strip=strip, survives=survives)
        e_del, e_det = self._apply(self.edges, select=select, strip=strip, survives=survives)
        return ProvenanceDeleteResult(
            nodes_deleted=n_del,
            edges_deleted=e_del,
            nodes_detached=n_det,
            edges_detached=e_det,
        )

    async def delete_by_dataset_id(self, dataset_id: UUID) -> ProvenanceDeleteResult:
        def select(a: _FakeArtifact) -> bool:
            return dataset_id in a.dataset_ids

        def strip(a: _FakeArtifact) -> None:
            a.dataset_ids.discard(dataset_id)

        # Detach only if the artifact still belongs to another dataset.
        def survives(a: _FakeArtifact) -> bool:
            return bool(a.dataset_ids)

        n_del, n_det = self._apply(self.nodes, select=select, strip=strip, survives=survives)
        e_del, e_det = self._apply(self.edges, select=select, strip=strip, survives=survives)
        return ProvenanceDeleteResult(
            nodes_deleted=n_del,
            edges_deleted=e_del,
            nodes_detached=n_det,
            edges_detached=e_det,
        )

    async def rollback_by_pipeline_run_id(
        self, pipeline_run_id: UUID, dataset_id: UUID
    ) -> ProvenanceDeleteResult:
        run_ref = make_source_run_ref(dataset_id, pipeline_run_id)

        def select(a: _FakeArtifact) -> bool:
            return run_ref in a.source_run_refs

        def strip(a: _FakeArtifact) -> None:
            a.source_run_refs.discard(run_ref)

        # Hard-delete only what this run *solely* introduced: no source ref and
        # no other source-run ref may keep the artifact alive.
        def survives(a: _FakeArtifact) -> bool:
            return bool(a.source_refs) or bool(a.source_run_refs)

        n_del, n_det = self._apply(self.nodes, select=select, strip=strip, survives=survives)
        e_del, e_det = self._apply(self.edges, select=select, strip=strip, survives=survives)
        return ProvenanceDeleteResult(
            nodes_deleted=n_del,
            edges_deleted=e_del,
            nodes_detached=n_det,
            edges_detached=e_det,
        )


# ---------------------------------------------------------------------------
# Fixtures / shared ids
# ---------------------------------------------------------------------------
DATASET_A = UUID("00000000-0000-0000-0000-0000000000a1")
DATASET_B = UUID("00000000-0000-0000-0000-0000000000b2")
DATA_1 = UUID("00000000-0000-0000-0000-0000000000d1")
DATA_2 = UUID("00000000-0000-0000-0000-0000000000d2")
RUN_OLD = UUID("00000000-0000-0000-0000-00000000ce01")
RUN_NEW = UUID("00000000-0000-0000-0000-00000000ce02")


# ---------------------------------------------------------------------------
# (1) deleting a source_ref that solely owns a node hard-deletes it
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_delete_source_ref_solely_owned_node_is_hard_deleted():
    store = FakeGraphVectorStore()
    ref = make_source_ref(DATASET_A, DATA_1)
    store.add_node("n1", source_refs=(ref,), dataset_ids=(DATASET_A,))

    result = await store.delete_by_source_ref(ref)

    assert "n1" not in store.nodes
    assert result == ProvenanceDeleteResult(nodes_deleted=1)
    assert result.nodes_deleted == 1
    assert result.nodes_detached == 0


# ---------------------------------------------------------------------------
# (2) deleting a source_ref on a node shared with another source_ref
#     detaches (survives) and increments nodes_detached
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_delete_source_ref_shared_node_is_detached_not_deleted():
    store = FakeGraphVectorStore()
    ref1 = make_source_ref(DATASET_A, DATA_1)
    ref2 = make_source_ref(DATASET_A, DATA_2)
    store.add_node("shared", source_refs=(ref1, ref2), dataset_ids=(DATASET_A,))

    result = await store.delete_by_source_ref(ref1)

    # Node survives because ref2 still owns it.
    assert "shared" in store.nodes
    assert store.nodes["shared"].source_refs == {ref2}
    assert result.nodes_detached == 1
    assert result.nodes_deleted == 0

    # Removing the last ref now hard-deletes it.
    result2 = await store.delete_by_source_ref(ref2)
    assert "shared" not in store.nodes
    assert result2.nodes_deleted == 1
    assert result2.nodes_detached == 0


# ---------------------------------------------------------------------------
# (3) delete_by_dataset_id removes all of one dataset's artifacts while
#     leaving another dataset's
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_delete_by_dataset_id_scopes_to_one_dataset():
    store = FakeGraphVectorStore()
    a_ref = make_source_ref(DATASET_A, DATA_1)
    b_ref = make_source_ref(DATASET_B, DATA_2)

    # Two nodes + one edge in dataset A.
    store.add_node("a_node_1", source_refs=(a_ref,), dataset_ids=(DATASET_A,))
    store.add_node("a_node_2", source_refs=(a_ref,), dataset_ids=(DATASET_A,))
    a_edge = EdgeIdentity("a_node_1", "knows", "a_node_2")
    store.add_edge(a_edge, source_refs=(a_ref,), dataset_ids=(DATASET_A,))

    # One node + one edge in dataset B.
    store.add_node("b_node_1", source_refs=(b_ref,), dataset_ids=(DATASET_B,))
    b_edge = EdgeIdentity("b_node_1", "rel", "b_node_1")
    store.add_edge(b_edge, source_refs=(b_ref,), dataset_ids=(DATASET_B,))

    result = await store.delete_by_dataset_id(DATASET_A)

    # Dataset A wiped.
    assert "a_node_1" not in store.nodes
    assert "a_node_2" not in store.nodes
    assert a_edge not in store.edges
    # Dataset B intact.
    assert "b_node_1" in store.nodes
    assert b_edge in store.edges

    assert result.nodes_deleted == 2
    assert result.edges_deleted == 1
    assert result.nodes_detached == 0
    assert result.edges_detached == 0


@pytest.mark.asyncio
async def test_delete_by_dataset_id_detaches_cross_dataset_artifact():
    """An artifact shared across two datasets survives a single-dataset delete."""
    store = FakeGraphVectorStore()
    a_ref = make_source_ref(DATASET_A, DATA_1)
    b_ref = make_source_ref(DATASET_B, DATA_2)
    store.add_node(
        "shared_across_ds",
        source_refs=(a_ref, b_ref),
        dataset_ids=(DATASET_A, DATASET_B),
    )

    result = await store.delete_by_dataset_id(DATASET_A)

    assert "shared_across_ds" in store.nodes
    assert store.nodes["shared_across_ds"].dataset_ids == {DATASET_B}
    assert result.nodes_detached == 1
    assert result.nodes_deleted == 0


# ---------------------------------------------------------------------------
# (4) rollback_by_pipeline_run_id removes only what that run solely
#     introduced and keeps artifacts a prior run created
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_rollback_removes_only_what_the_run_solely_introduced():
    store = FakeGraphVectorStore()
    old_run_ref = make_source_run_ref(DATASET_A, RUN_OLD)
    new_run_ref = make_source_run_ref(DATASET_A, RUN_NEW)

    # Node created only by the new run -> rolled back hard-deletes it.
    store.add_node("only_new", source_run_refs=(new_run_ref,), dataset_ids=(DATASET_A,))

    # Node created by a prior run, untouched by the new run -> survives.
    store.add_node("prior_only", source_run_refs=(old_run_ref,), dataset_ids=(DATASET_A,))

    # Node touched by both runs -> new run's ref stripped but old run keeps it.
    store.add_node(
        "touched_by_both",
        source_run_refs=(old_run_ref, new_run_ref),
        dataset_ids=(DATASET_A,),
    )

    # Node that has a surviving source ref despite the new run -> detached.
    src_ref = make_source_ref(DATASET_A, DATA_1)
    store.add_node(
        "has_source_ref",
        source_refs=(src_ref,),
        source_run_refs=(new_run_ref,),
        dataset_ids=(DATASET_A,),
    )

    result = await store.rollback_by_pipeline_run_id(RUN_NEW, DATASET_A)

    # Solely-introduced node gone.
    assert "only_new" not in store.nodes
    # Prior run's node untouched (its run ref still present, new ref absent).
    assert "prior_only" in store.nodes
    assert store.nodes["prior_only"].source_run_refs == {old_run_ref}
    # Both-runs node survives, only the new run's ref stripped.
    assert "touched_by_both" in store.nodes
    assert store.nodes["touched_by_both"].source_run_refs == {old_run_ref}
    # Node with a live source ref survives the rollback.
    assert "has_source_ref" in store.nodes
    assert store.nodes["has_source_ref"].source_run_refs == set()
    assert store.nodes["has_source_ref"].source_refs == {src_ref}

    # 1 hard-deleted (only_new); 2 detached (touched_by_both, has_source_ref).
    # prior_only was never selected (no new run ref), so it isn't counted.
    assert result.nodes_deleted == 1
    assert result.nodes_detached == 2


@pytest.mark.asyncio
async def test_rollback_uses_dataset_scoped_run_ref():
    """The same run id in a different dataset yields a different ref and is a no-op."""
    store = FakeGraphVectorStore()
    run_ref_a = make_source_run_ref(DATASET_A, RUN_NEW)
    store.add_node("a_only", source_run_refs=(run_ref_a,), dataset_ids=(DATASET_A,))

    # Roll back the same run id but scoped to dataset B -> different ref -> no-op.
    result = await store.rollback_by_pipeline_run_id(RUN_NEW, DATASET_B)

    assert "a_only" in store.nodes
    assert result == ProvenanceDeleteResult()


# ---------------------------------------------------------------------------
# (5) the un-overridden GraphVectorStoreInterface base still raises
#     UnsupportedProvenanceCapability for all three methods
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_base_interface_raises_unsupported_for_all_methods():
    base = GraphVectorStoreInterface()

    assert base.supports_graph_native_delete() is False

    with pytest.raises(UnsupportedProvenanceCapability) as exc1:
        await base.delete_by_source_ref(make_source_ref(DATASET_A, DATA_1))
    assert exc1.value.capability == "delete_by_source_ref"

    with pytest.raises(UnsupportedProvenanceCapability) as exc2:
        await base.delete_by_dataset_id(DATASET_A)
    assert exc2.value.capability == "delete_by_dataset_id"

    with pytest.raises(UnsupportedProvenanceCapability) as exc3:
        await base.rollback_by_pipeline_run_id(RUN_NEW, DATASET_A)
    assert exc3.value.capability == "rollback_by_pipeline_run_id"

    # Subclasses NotImplementedError so legacy handlers keep catching it.
    with pytest.raises(NotImplementedError):
        await base.delete_by_source_ref(make_source_ref(DATASET_A, DATA_1))


# ---------------------------------------------------------------------------
# sanity: the fake reports it supports the capability
# ---------------------------------------------------------------------------
def test_fake_supports_graph_native_delete():
    assert FakeGraphVectorStore().supports_graph_native_delete() is True
