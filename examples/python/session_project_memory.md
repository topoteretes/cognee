# Project-scoped session memory

Send `node_set: ["project-<canonical-path-hash>"]` inside a typed QA or trace entry
to `POST /api/v1/remember/entry`, or from the SDK in either spelling:

```python
await cognee.remember(
    cognee.QAEntry(question=q, answer=a, node_set=["project-<hash>"]), session_id="s1"
)
await cognee.remember(cognee.QAEntry(question=q, answer=a), session_id="s1", node_set=["project-<hash>"])
await cognee.remember("a note", session_id="s1", node_set=["project-<hash>"])  # plain text pins too
```

`node_set` takes the same node-set names `add()` does, and like the call-level
`node_set` on `remember()`, `add()` and `update()` it is not size-limited; only an
empty name is refused. The first write that carries one pins it on the session;
later writes must repeat it or omit it, and a different set for the same
authenticated user and session is rejected with HTTP 409
(`cognee.SessionNodeSetConflictError` in the SDK; the message names both sets). An
empty list means "no node set" and pins nothing, so a client can start without one
and pin it later. Giving the field and the kwarg different values on one call is a
`ValueError`; the kwarg on an entry type that cannot carry a node set (feedback,
skill runs) is a `TypeError`.

`improve(session_ids=[...])` keeps the pinned set on everything it bridges from the
session into the graph: the persisted Q&A (`user_sessions_from_cache`), the
persisted agent traces (`agent_trace_feedbacks`) and the distilled session lessons
(`session_learnings`) each get the session's node set appended to their own. A
project-scoped read is then an ordinary recall filtered by node set:

```python
await cognee.recall(
    "how do we run the migrations here?",
    datasets=["main_dataset"],
    node_name=["project-<canonical-path-hash>", "global"],
    node_name_filter_operator="OR",
)
```

User preferences get no node set on purpose: they belong to the user and stay
recallable from every project.

The pin is session state: it is read strictly (an unreadable session is never
treated as unpinned, so a pin is refused rather than overwritten, and the bridging
stages skip that session for the run instead of adding it untagged), and it goes
away with the session. The pin is taken under the session-turn lock. A single
worker needs nothing extra; multiple workers need the same distributed-lock
configuration as every other session read-modify-write.

Over MCP, the `remember` tool takes `node_set` and forwards it on every path: the
typed entry's field in API session mode, form fields in API permanent mode, the
call-level kwarg in local mode.

The end-to-end contract (two sessions, two sets, improve, scoped recall keeps them
apart) is pinned by `test_session_node_set_scopes_what_improve_bridges_into_the_graph`
in `cognee/tests/journeys/test_session_journey.py`, and the wire contract by
`test_http_session_entries_pin_a_node_set` in `test_http_api_journey.py`.
