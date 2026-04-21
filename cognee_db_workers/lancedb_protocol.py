"""Op-codes exchanged between the LanceDB subprocess worker and the main-side
proxies. Pure stdlib.
"""

from __future__ import annotations

OP_CONNECT = 100              # kwargs: url, api_key

OP_TABLE_NAMES = 110          # no args; returns list[str]
OP_CREATE_TABLE = 111         # args: (name, schema_bytes, exist_ok); schema is pickled pa.Schema
OP_OPEN_TABLE = 112           # args: (name,); returns handle
OP_DROP_TABLE = 113           # args: (name,)

OP_TABLE_COUNT_ROWS = 120     # handle_id
OP_TABLE_TO_ARROW = 121       # handle_id; returns pickled pa.Table (via bytes)
OP_TABLE_ADD = 122            # handle_id; args: (records: list[dict])
OP_TABLE_DELETE = 123         # handle_id; args: (where: str)
OP_TABLE_RELEASE = 124        # handle_id; release the table handle (no-op if already gone)

# Builder ops. args: (chain_steps, terminal_name, terminal_args, terminal_kwargs)
# where chain_steps = list[(method_name, args, kwargs)] to apply on top of the
# initial builder.
OP_TABLE_QUERY_EXECUTE = 130            # root = table.query()
OP_TABLE_VECTOR_SEARCH_EXECUTE = 131    # root = table.vector_search(vec)
OP_TABLE_MERGE_INSERT_EXECUTE = 132     # root = table.merge_insert(key)
