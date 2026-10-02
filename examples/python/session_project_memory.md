# Project-scoped session memory

Send `node_set: ["project-<canonical-path-hash>"]` inside a typed QA or trace entry
to `POST /api/v1/remember/entry` (or `remember(QAEntry(..., node_set=[...]),
session_id=...)` in the SDK). The first tagged entry pins that set on the session;
later entries must repeat it or omit it, and a different set for the same
authenticated user and session is rejected with HTTP 409 (`ProjectTagConflictError`
in the SDK). An empty list means "no tags" and pins nothing, so a client can start
untagged and tag later.

`improve(session_ids=[...])` keeps the pinned tags on everything it bridges from
the session into the graph: the persisted Q&A (`user_sessions_from_cache`), the
persisted agent traces (`agent_trace_feedbacks`) and the distilled session lessons
(`session_learnings`) each get the project tags appended to their node set. A
project-scoped read is then an ordinary recall filtered by node set:

```python
await cognee.recall(
    "how do we run the migrations here?",
    datasets=["main_dataset"],
    node_name=["project-<canonical-path-hash>", "global"],
    node_name_filter_operator="OR",
)
```

User preferences are not tagged on purpose: they belong to the user and stay
recallable from every project.

The pin is taken under the session-turn lock. A single worker needs nothing
extra; multiple workers need the same distributed-lock configuration as every
other session read-modify-write.
