# Choose the accepted fact in a conflict

Conflict review (`improve(review_conflicts=True)`) creates `FactConflict` nodes.
A user with dataset write permission can choose one of a conflict's facts.
The choice marks the conflict `resolved`, records who made it, when, and why,
and survives subsequent automatic reviews while the selected fact still exists.
If its supporting document is forgotten and the fact disappears, the next conflict
review replaces or drops the decision and clears its obsolete rejection marks. A
forgotten competing fact leaves no trace in the decision: its id, text, value and
cited chunks drop out at the next review.
A decision covers only the facts listed when it was made: a competing fact that
a later review finds in new data is not marked rejected on the user's behalf. It
keeps the review's status and is listed next to the decision, so the choice can
be revised with it in view. Choosing a different fact explicitly revises the
decision. Repeating the same choice over the same facts is idempotent; repeated
after a review listed new evidence, it rejects that evidence too.

Inspect the candidates:

```http
GET /api/v1/conflicts?dataset_id=<dataset UUID>&status=unresolved
Authorization: Bearer <token>
```

The response includes `facts`, each with an `id`, `text`, status, and source
citations. Use one of those fact ids, rather than an entity id or free-form text:

```http
POST /api/v1/conflicts/<conflict UUID>/resolve
Authorization: Bearer <token>
Content-Type: application/json

{
  "datasetId": "<dataset UUID>",
  "selectedFactId": "<fact UUID from the list>",
  "reason": "Confirmed with the document owner"
}
```

Read permission is required for inspection; write permission is required for
resolution. A conflict outside the requested dataset is not exposed. A fact
that does not belong to the conflict is rejected with HTTP 422.

The Python SDK exposes the same operations, locally or, after `cognee.serve()`,
against the connected instance:

```python
conflicts = await cognee.list_conflicts(dataset_id, status="unresolved")
result = await cognee.resolve_conflict(
    conflict_id,
    dataset_id=dataset_id,
    selected_fact_id=selected_fact_id,
    reason="Confirmed with the document owner",
)
```

Normal graph, hybrid, chunk, lexical, and summary retrieval uses the accepted
information. Passages and summaries keep their full source text and carry
`resolved_facts` plus an explicit annotation giving the selected fact precedence
over competing claims. Unrelated graph edges and node descriptions remain intact;
only explicitly rejected fact edges and standalone fact hits are mapped to the
accepted fact without waiting for re-embedding. Graph context and hybrid fact
lists label the selected fact `[accepted by user]`, and a rejected edge retrieved
together with it is dropped, so the edges of one retrieval round state each
decision once. Entity and passage annotations, and the optional prelude, restate
the decision. These changes affect retrieved copies only: original documents and
graph facts remain available for inspection and raw graph queries. Cached answers
and lessons with recorded references to a competing fact, the conflict, or a
passage cited for it are invalidated; answers that only mention the subject entity
are kept.

The optional global-context prelude keeps its root and top-k summaries. It appends
every user decision whose subject, values, or sources intersect the retrieved
nodes, and no other, leaving out a decision whose selected fact is already in the
context. Reading decisions is required for normal retrieval; a graph
failure surfaces an error rather than returning an unchecked old hit.
Custom retrievers and raw graph queries must apply the same resolution policy
if they are used to generate answers. Cache cleanup relies on recorded fact
references; old untracked turns and append-only cache adapters cannot be cleaned
by this mechanism.

Resolution protects graph writes with the same per-process dataset lock as
ingestion and improve. Cache cleanup runs outside that lock, with bounded
concurrency across sessions; the decision remains `write_pending` until cleanup
succeeds. Finalization checks for a newer decision or a deleted conflict before
writing. Failed writes or cleanup can be retried with the same request. As with
ingestion, applications with several API workers must route writes for a dataset
through one worker.

See [the runnable example](../examples/guides/resolve_conflicts.py).
