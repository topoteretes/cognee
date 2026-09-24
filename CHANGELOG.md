# Changelog

## Unreleased

### Added

- Streamed `GET /api/v1/visualize/json`: with `Accept: text/event-stream` or `stream=true` the
  graph arrives as server-sent events, `meta`, then one `chunk` per read (compact nodes and links,
  no property bag), then `summary` events with `importance`, `label_priority` and
  `color_maps.node_set`, then `done`. Every link's endpoints are sent before it. `max_nodes` goes
  up to 20000 when streamed; the JSON response keeps its 5000 cap and rejects more exactly as
  before, and `full=true` cannot be streamed. A process holds at most 8 streams at once and
  answers 503 beyond that. Without opting in the JSON response is unchanged (SDK-787).

- `GraphDBInterface.iter_bounded_neighborhood(node_ids, depth, max_nodes, chunk_size=2000,
  property_keys=None)`: the seeds' neighbourhood capped at `max_nodes`, yielded in chunks.
  Seeds come first, then nodes by hop distance; every edge between two members arrives
  exactly once, in the chunk of its later endpoint, so no chunk refers to a node not yet
  received. `property_keys` limits node properties to those keys plus `name` and `type`.
  It is not abstract: adapters without a native implementation inherit a default built on
  `get_neighborhood`, which still reads the whole neighbourhood and logs a warning once per
  adapter type. `postgres_demo` implements it natively: the traversal stops at `max_nodes`
  in SQL, each hop's budget is shared round robin across its frontier, and node rows are
  read one chunk at a time (SDK-786).

### Changed

- Graph visualization (`visualize_graph`, `/api/v1/visualize`, `/api/v1/visualize/json`)
  reads the seed neighbourhood through `iter_bounded_neighborhood` instead of reading the
  whole neighbourhood with `get_neighborhood` and cutting it to `max_nodes` in Python.
  Returned nodes are now ordered seeds first, then by hop distance. On adapters with a
  native implementation, which nodes of the last admitted hop are kept can differ from
  before, since the budget is shared across the frontier (SDK-786).

- `GET /api/v1/datasets/{id}/data` now returns up to 100 documents by default.
  Clients must follow `limit` (1–1000) and `offset` (0–1,000,000) to read additional
  pages. The array response has no `X-Total-Count` header or truncation marker;
  use `GET /api/v1/datasets/{id}/data/count` to obtain the total. SDK `list_data`
  and explicit traversal clients follow pages and deduplicate overlaps. Offset
  traversal is not a snapshot: concurrent inserts/deletes may omit rows.
  The count is uncapped, but pages beyond offset 1,000,000 are rejected; datasets
  above 1,001,000 documents cannot be fully traversed through this endpoint.
  Full-traversal clients surface that error instead of returning partial results.
