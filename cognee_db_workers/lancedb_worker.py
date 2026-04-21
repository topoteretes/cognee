"""LanceDB subprocess worker. Imports only ``lancedb`` + ``pyarrow`` + stdlib
+ harness/protocol. Must not import cognee.
"""

from __future__ import annotations

import pickle

from .harness import HandleRegistry, HandleResult, Request, run_worker_loop
from .lancedb_protocol import (
    OP_CONNECT,
    OP_CREATE_TABLE,
    OP_DROP_TABLE,
    OP_OPEN_TABLE,
    OP_TABLE_ADD,
    OP_TABLE_COUNT_ROWS,
    OP_TABLE_DELETE,
    OP_TABLE_MERGE_INSERT_EXECUTE,
    OP_TABLE_NAMES,
    OP_TABLE_QUERY_EXECUTE,
    OP_TABLE_RELEASE,
    OP_TABLE_TO_ARROW,
    OP_TABLE_VECTOR_SEARCH_EXECUTE,
)


# The connection is stored at a fixed handle id (0) since there is exactly one
# per worker.
_CONNECTION_HANDLE = 0


def _get_connection(registry: HandleRegistry):
    return registry.get(_CONNECTION_HANDLE)


async def _op_connect(registry: HandleRegistry, req: Request) -> None:
    import lancedb

    url = req.kwargs["url"]
    api_key = req.kwargs.get("api_key")

    connection = await lancedb.connect_async(url, api_key=api_key)
    registry._handles[_CONNECTION_HANDLE] = connection  # fixed slot
    return None


async def _op_table_names(registry: HandleRegistry, req: Request):
    conn = _get_connection(registry)
    return await conn.table_names()


def _relax_nullability(schema):
    """Return a pa.Schema with every top-level field (and nested struct fields)
    marked as nullable. LanceModel's ``to_arrow_schema()`` marks required
    pydantic fields as non-nullable; once pydantic validation happens only in
    the main process, the worker sees plain dicts and some records may end up
    with null values (e.g. optional-ish fields that pydantic would have
    defaulted). Relaxing nullability avoids brittle schema mismatches at the
    Arrow layer.
    """
    import pyarrow as pa

    def _relax_type(arrow_type):
        if pa.types.is_struct(arrow_type):
            return pa.struct(
                [pa.field(f.name, _relax_type(f.type), nullable=True) for f in arrow_type]
            )
        if pa.types.is_list(arrow_type):
            return pa.list_(
                pa.field(arrow_type.value_field.name, _relax_type(arrow_type.value_type), nullable=True)
            )
        if pa.types.is_fixed_size_list(arrow_type):
            return pa.list_(
                pa.field(
                    arrow_type.value_field.name,
                    _relax_type(arrow_type.value_type),
                    nullable=True,
                ),
                arrow_type.list_size,
            )
        return arrow_type

    return pa.schema([pa.field(f.name, _relax_type(f.type), nullable=True) for f in schema])


async def _op_create_table(registry: HandleRegistry, req: Request):
    import pyarrow as pa  # noqa: F401  # ensure pyarrow is resolved in-worker

    conn = _get_connection(registry)
    name = req.args[0]
    schema_bytes = req.args[1]
    exist_ok = bool(req.args[2]) if len(req.args) > 2 else True

    schema = pickle.loads(schema_bytes)
    if isinstance(schema, pa.Schema):
        schema = _relax_nullability(schema)
    await conn.create_table(name=name, schema=schema, exist_ok=exist_ok)
    return None


async def _op_open_table(registry: HandleRegistry, req: Request) -> HandleResult:
    conn = _get_connection(registry)
    name = req.args[0]
    table = await conn.open_table(name)
    return HandleResult(value=None, handle_id=registry.register(table))


async def _op_drop_table(registry: HandleRegistry, req: Request):
    conn = _get_connection(registry)
    name = req.args[0]
    await conn.drop_table(name)
    return None


def _op_release_handle(registry: HandleRegistry, req: Request):
    """Drop a handle from the registry. Idempotent."""
    if req.handle_id is not None:
        registry.pop(req.handle_id)
    return None


async def _op_table_count_rows(registry: HandleRegistry, req: Request):
    table = registry.get(req.handle_id)
    return await table.count_rows()


async def _op_table_to_arrow(registry: HandleRegistry, req: Request):
    table = registry.get(req.handle_id)
    arrow = await table.to_arrow()
    # Serialize the arrow Table via pyarrow's IPC stream for robust transfer.
    import pyarrow as pa

    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, arrow.schema) as writer:
        writer.write_table(arrow)
    return sink.getvalue().to_pybytes()


async def _op_table_add(registry: HandleRegistry, req: Request):
    table = registry.get(req.handle_id)
    records = req.args[0]
    await table.add(records)
    return None


async def _op_table_delete(registry: HandleRegistry, req: Request):
    table = registry.get(req.handle_id)
    where_expr = req.args[0]
    await table.delete(where_expr)
    return None


def _apply_chain(builder, chain_steps):
    for method_name, args, kwargs in chain_steps:
        builder = getattr(builder, method_name)(*args, **kwargs)
    return builder


async def _run_builder(table, root_args, chain_steps, terminal_name, terminal_args, terminal_kwargs, root_method: str):
    builder = getattr(table, root_method)(*root_args)
    builder = _apply_chain(builder, chain_steps)
    terminal = getattr(builder, terminal_name)(*terminal_args, **terminal_kwargs)
    # Some terminal methods are awaitable, some are not. LanceDB async API
    # terminals we care about (to_list, execute) are awaitable.
    import inspect

    if inspect.iscoroutine(terminal) or inspect.isawaitable(terminal):
        return await terminal
    return terminal


async def _op_query_execute(registry: HandleRegistry, req: Request):
    table = registry.get(req.handle_id)
    root_args, chain, terminal_name, t_args, t_kwargs = req.args
    return await _run_builder(
        table, root_args, chain, terminal_name, t_args, t_kwargs, "query"
    )


async def _op_vector_search_execute(registry: HandleRegistry, req: Request):
    table = registry.get(req.handle_id)
    root_args, chain, terminal_name, t_args, t_kwargs = req.args
    return await _run_builder(
        table, root_args, chain, terminal_name, t_args, t_kwargs, "vector_search"
    )


async def _op_merge_insert_execute(registry: HandleRegistry, req: Request):
    table = registry.get(req.handle_id)
    root_args, chain, terminal_name, t_args, t_kwargs = req.args
    result = await _run_builder(
        table, root_args, chain, terminal_name, t_args, t_kwargs, "merge_insert"
    )
    # lancedb's MergeResult is a Rust-backed pyo3 object that isn't picklable.
    # The adapter only cares that execute() returned without error, so we
    # reduce the result to a plain dict of its counters when available.
    if result is None:
        return None
    try:
        return {
            "num_inserted_rows": getattr(result, "num_inserted_rows", None),
            "num_updated_rows": getattr(result, "num_updated_rows", None),
            "num_deleted_rows": getattr(result, "num_deleted_rows", None),
        }
    except Exception:
        return None


DISPATCH = {
    OP_CONNECT: _op_connect,
    OP_TABLE_NAMES: _op_table_names,
    OP_CREATE_TABLE: _op_create_table,
    OP_OPEN_TABLE: _op_open_table,
    OP_DROP_TABLE: _op_drop_table,
    OP_TABLE_RELEASE: _op_release_handle,
    OP_TABLE_COUNT_ROWS: _op_table_count_rows,
    OP_TABLE_TO_ARROW: _op_table_to_arrow,
    OP_TABLE_ADD: _op_table_add,
    OP_TABLE_DELETE: _op_table_delete,
    OP_TABLE_QUERY_EXECUTE: _op_query_execute,
    OP_TABLE_VECTOR_SEARCH_EXECUTE: _op_vector_search_execute,
    OP_TABLE_MERGE_INSERT_EXECUTE: _op_merge_insert_execute,
}


def worker_main(req_q, resp_q) -> None:
    run_worker_loop(DISPATCH, req_q, resp_q)
