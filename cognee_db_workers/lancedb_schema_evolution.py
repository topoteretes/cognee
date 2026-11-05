"""In-place payload schema evolution for LanceDB tables, shared by the
in-process adapter and the subprocess worker. Pure pyarrow plus the pylance
dataset it is handed. Must not import cognee.

A cognee table has three columns: ``id``, ``vector`` and a ``payload`` struct.
When the payload model changes (a field added or removed, a type or a
nullability change) the stored struct no longer matches the one the running
code writes. The vectors never change in such a migration, so the ``payload``
column is evolved on its own with Lance's schema evolution:

1. ``add_columns`` computes a new struct column, ``__payload_migrating``, from
   the old one with a Python UDF that reads only ``payload``, batch by batch.
   Kept fields are cast to their new type where it changed, new fields take
   the model's default (or null when the field may be null), fields the model
   no longer has are left out.
2. ``alter_columns`` renames ``payload`` to ``__payload_old`` and
   ``__payload_migrating`` to ``payload`` in one commit.
3. ``drop_columns`` removes ``__payload_old``.

Each step is a committed Lance version and ``payload`` exists at every one of
them, so a reader never sees a table without it and an interrupted migration
leaves the table readable. The temporary columns are nullable (the swap also
makes the old column nullable), so a write that carries only ``id``,
``vector`` and ``payload`` still lands between the steps: a non-nullable
column the writer does not provide would be refused by ``merge_insert``. The
next run finishes it: a leftover ``__payload_old`` is dropped (the swap
committed, ``payload`` is the new struct), a leftover ``__payload_migrating``
is dropped and computed again. The vector column's data files are never
rewritten, and no row is ever materialized as Python objects.

Nothing partial: a required target field that is missing from the stored
struct with no default, a value that cannot be cast to its new type, or a null
in a required field that no default can fill, fails the UDF, and
``add_columns`` commits nothing.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc

#: Where the evolved struct is built before it replaces ``payload``.
MIGRATING_COLUMN = "__payload_migrating"
#: Where the previous struct sits between the swap and its drop.
OLD_PAYLOAD_COLUMN = "__payload_old"
PAYLOAD_COLUMN = "payload"

_NO_DEFAULT = object()


_ABORTED = "LanceDB payload schema migration aborted"


class SchemaEvolutionError(RuntimeError):
    """The stored payload cannot be brought to the target schema; nothing was changed."""

    def __init__(self, reason: str):
        super().__init__(f"{_ABORTED}: {reason}")
        self.reason = reason

    def __reduce__(self):
        # Pickled across the worker boundary: rebuild from the reason, not the
        # formatted message, or the prefix would be added twice.
        return (SchemaEvolutionError, (self.reason,))


def relax_nullability(schema: pa.Schema) -> pa.Schema:
    """``schema`` with every field, nested struct and list fields included, nullable.

    ``LanceModel.to_arrow_schema()`` marks required pydantic fields non-nullable.
    The subprocess worker creates every table with this relaxed form: it sees
    plain dicts, never pydantic-validated rows, so a record may carry a null
    pydantic would have defaulted. The adapter compares stored schemas against
    the same relaxed form in that mode. Which fields must hold a value is the
    model's rule, not the storage schema's: a migration gets it as ``required``.
    """

    def _relax_type(arrow_type):
        if pa.types.is_struct(arrow_type):
            return pa.struct(
                [pa.field(f.name, _relax_type(f.type), nullable=True) for f in arrow_type]
            )
        if pa.types.is_list(arrow_type):
            return pa.list_(
                pa.field(
                    arrow_type.value_field.name, _relax_type(arrow_type.value_type), nullable=True
                )
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


@dataclass(frozen=True)
class FieldStep:
    """How one target field is produced from the stored struct."""

    name: str
    #: ``keep`` (same type), ``cast`` (stored under another type), ``default``
    #: (missing from the stored struct, filled with ``default``), ``null``
    #: (missing, may be null, no default).
    source: str
    default: Any = _NO_DEFAULT
    #: Every row must hold a value: the model requires the field, or the
    #: storage schema declares it non-null.
    required: bool = False


def arrow_default(value: Any) -> Any:
    """A model default as a plain value pyarrow can build an array from.

    Plain means picklable without importing the model's module: the adapter
    applies this before the defaults cross to the worker. Enum members become
    their value, pydantic models their ``model_dump``, tuples and sets lists;
    datetimes, dates, times and decimals pyarrow takes as they are; anything
    else (a UUID, a path) its string form.
    """
    if isinstance(value, Enum):
        return arrow_default(value.value)
    if hasattr(value, "model_dump"):
        return arrow_default(value.model_dump())
    if isinstance(value, (tuple, set, frozenset, list)):
        return [arrow_default(item) for item in value]
    if isinstance(value, dict):
        return {key: arrow_default(item) for key, item in value.items()}
    if value is None or isinstance(
        value, (str, int, float, bool, bytes, datetime, date, time, timedelta, Decimal)
    ):
        return value
    return str(value)


def _default_array(field: pa.Field, default: Any, row_count: int) -> pa.Array:
    """``row_count`` copies of ``default`` as ``field``'s type, or ``SchemaEvolutionError``."""
    try:
        return pa.array([arrow_default(default)] * row_count, type=field.type)
    except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError) as error:
        raise SchemaEvolutionError(
            f"the default {default!r} of field '{field.name}' is not a {field.type} ({error}). "
            "No data was modified. Give the field a default of its own type and re-run "
            "migration."
        ) from error


def plan_payload_evolution(
    stored: pa.StructType,
    target: pa.StructType,
    defaults: dict[str, Any],
    required: Iterable[str],
) -> list[FieldStep]:
    """One step per target field, or ``SchemaEvolutionError`` when a field cannot be produced.

    ``required`` names the fields the model requires a value for. It is kept
    apart from the target's nullability because the subprocess worker stores
    every field nullable (``relax_nullability``) while the model still refuses
    a null in a required field.
    """
    required = frozenset(required)
    stored_names = {stored.field(i).name: stored.field(i) for i in range(stored.num_fields)}
    steps: list[FieldStep] = []
    missing_without_default: list[str] = []
    for field in target:
        must_hold_value = field.name in required or not field.nullable
        default = defaults.get(field.name, _NO_DEFAULT)
        existing = stored_names.get(field.name)
        if existing is not None:
            source = "keep" if existing.type == field.type else "cast"
            steps.append(FieldStep(field.name, source, default, must_hold_value))
        elif default is not _NO_DEFAULT:
            steps.append(FieldStep(field.name, "default", default, must_hold_value))
        elif not must_hold_value:
            steps.append(FieldStep(field.name, "null"))
        else:
            missing_without_default.append(field.name)
    if missing_without_default:
        raise SchemaEvolutionError(
            f"the stored rows have no value for {', '.join(missing_without_default)} and the "
            "model gives none. No data was modified. Add an explicit default value for newly "
            "required fields (or make them optional) and re-run migration."
        )
    return steps


def _build_struct(payload: pa.Array, stored: pa.StructType, target, steps) -> pa.StructArray:
    """The target struct for one batch of stored payloads."""
    row_count = len(payload)
    children = payload.flatten()
    arrays = []
    for field, step in zip(target, steps):
        if step.source in ("keep", "cast"):
            array = children[stored.get_field_index(field.name)]
            if step.source == "cast":
                try:
                    array = array.cast(field.type)
                except (pa.ArrowInvalid, pa.ArrowTypeError, pa.ArrowNotImplementedError) as error:
                    raise SchemaEvolutionError(
                        f"field '{field.name}' cannot be cast from {array.type} to "
                        f"{field.type} ({error}). No data was modified."
                    ) from error
        elif step.source == "default":
            array = _default_array(field, step.default, row_count)
        else:
            array = pa.nulls(row_count, field.type)

        if step.required and array.null_count:
            if step.default is _NO_DEFAULT or step.default is None:
                raise SchemaEvolutionError(
                    f"field '{field.name}' is required by the new schema but some stored rows "
                    "hold null, and the model gives no default to fill them. No data was "
                    "modified. Add an explicit default value (or make the field optional) and "
                    "re-run migration."
                )
            array = pc.fill_null(array, _default_array(field, step.default, 1)[0])
        arrays.append(array)
    return pa.StructArray.from_arrays(arrays, fields=list(target))


def _finish_interrupted(dataset) -> None:
    """Bring a table an earlier migration left mid-way back to the start state.

    Only two intermediate states exist, since ``payload`` is present at every
    committed version: after the swap (``__payload_old`` still there: the new
    struct is already ``payload``, drop the old one), or after the first step
    (``__payload_migrating`` next to ``payload``: drop it, it is computed again
    against the current target).
    """
    names = dataset.schema.names
    if PAYLOAD_COLUMN not in names:
        raise SchemaEvolutionError(
            f"the table has no '{PAYLOAD_COLUMN}' column (columns: {', '.join(names)}). "
            "No data was modified."
        )
    leftovers = [name for name in (OLD_PAYLOAD_COLUMN, MIGRATING_COLUMN) if name in names]
    if leftovers:
        dataset.drop_columns(leftovers)


def evolve_payload_schema(
    dataset,
    target: pa.StructType,
    defaults: dict[str, Any],
    *,
    required: Iterable[str],
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Bring ``dataset``'s ``payload`` column to ``target`` in place; see the module docstring.

    ``dataset`` is a pylance ``LanceDataset`` at the table's latest version.
    ``defaults`` maps target field names to the model's default values and
    ``required`` names the fields the model requires a value for.
    Returns what changed: the fields added, removed and cast, and the row count.
    """
    _finish_interrupted(dataset)

    stored = dataset.schema.field(PAYLOAD_COLUMN).type
    if not pa.types.is_struct(stored):
        raise SchemaEvolutionError(
            f"'{PAYLOAD_COLUMN}' is {stored}, not a struct. No data was modified."
        )
    if stored == target:
        # A migration interrupted after its swap: the payload is already the
        # target, only the old column was left to drop, done above.
        return {"rows": dataset.count_rows(), "added": [], "removed": [], "cast": []}
    steps = plan_payload_evolution(stored, target, defaults, required)
    target_names = {field.name for field in target}

    from lance.udf import batch_udf

    # An explicit output schema: without one Lance probes the callable on a
    # sample batch and reports any error as a TypeError about the transform.
    # Nullable, so a write between the steps (which does not carry this
    # column) is not refused; every row the UDF produces holds a struct.
    output = pa.schema([pa.field(MIGRATING_COLUMN, target, nullable=True)])

    @batch_udf(output_schema=output)
    def transform(batch: pa.RecordBatch) -> pa.RecordBatch:
        struct = _build_struct(batch.column(PAYLOAD_COLUMN), stored, target, steps)
        return pa.RecordBatch.from_arrays([struct], schema=output)

    try:
        dataset.add_columns(transform, read_columns=[PAYLOAD_COLUMN], batch_size=batch_size)
    except Exception as error:
        # Lance reports an exception raised inside the UDF as an OSError carrying
        # its text; nothing was committed. Surface ours under its own type.
        if _ABORTED in str(error):
            raise SchemaEvolutionError(str(error).split(_ABORTED, 1)[1].strip(" :,\n")) from error
        raise
    # One commit: the table goes from the old ``payload`` to the new one with
    # no version in between that lacks the column. The old column turns
    # nullable in the same commit, for the writes that land before its drop.
    dataset.alter_columns(
        {"path": PAYLOAD_COLUMN, "name": OLD_PAYLOAD_COLUMN, "nullable": True},
        {"path": MIGRATING_COLUMN, "name": PAYLOAD_COLUMN},
    )
    dataset.drop_columns([OLD_PAYLOAD_COLUMN])

    return {
        "rows": dataset.count_rows(),
        "added": [step.name for step in steps if step.source in ("default", "null")],
        "removed": [
            stored.field(i).name
            for i in range(stored.num_fields)
            if stored.field(i).name not in target_names
        ],
        "cast": [step.name for step in steps if step.source == "cast"],
    }
