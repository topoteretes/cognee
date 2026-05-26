"""Op-codes exchanged between the LanceDB subprocess worker and the main-side
proxies. Pure stdlib.
"""

from __future__ import annotations

OP_CONNECT = 100  # kwargs: url, api_key

OP_TABLE_NAMES = 110  # no args; returns list[str]
# args: (name, schema_bytes, exist_ok). ``schema_bytes`` is Arrow IPC
# serialized (``schema.serialize().to_pybytes()`` on the proxy side,
# ``pa.ipc.read_schema`` on the worker side). NOT pickled — pickle.loads
# on a subprocess RPC is an RCE surface, Arrow IPC is a typed format that
# rejects non-schema bytes.
OP_CREATE_TABLE = 111
OP_OPEN_TABLE = 112  # args: (name,); returns handle
OP_DROP_TABLE = 113  # args: (name,)

OP_TABLE_COUNT_ROWS = 120  # handle_id
OP_TABLE_TO_ARROW = 121  # handle_id; returns pa.Table serialized as IPC stream bytes
# handle_id; args: (records,) — accepts whatever lancedb's AsyncTable.add
# accepts. In subprocess mode the cognee adapter sends a pa.Table built by
# ``LanceDBAdapter._records_for_write`` (so the worker never has to import
# pydantic). list[dict] / list[pa.RecordBatch] / pa.RecordBatchReader also
# work because lancedb itself accepts those.
OP_TABLE_ADD = 122
OP_TABLE_DELETE = 123  # handle_id; args: (where: str)
OP_TABLE_RELEASE = 124  # handle_id; release the table handle (no-op if already gone)

# Builder ops. args: (root_args, chain_steps, terminal_name, terminal_args,
# terminal_kwargs) where root_args is the tuple passed to the root call
# (e.g. ``(vector,)`` for ``vector_search``) and chain_steps is a
# ``list[(method_name, args, kwargs)]`` of fluent calls applied on top of
# the initial builder.
OP_TABLE_QUERY_EXECUTE = 130  # root = table.query()
OP_TABLE_VECTOR_SEARCH_EXECUTE = 131  # root = table.vector_search(vec)
OP_TABLE_MERGE_INSERT_EXECUTE = 132  # root = table.merge_insert(key)
