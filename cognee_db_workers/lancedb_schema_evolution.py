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
3. Rows another process wrote between steps 1 and 2 (the adapter's lock covers
   this process only) have their real payload in ``__payload_old`` and a null
   or stale ``payload``: Lance refuses step 1's commit over a concurrent
   write, but rebases step 2 over one. They are exactly the rows of the
   fragments present at the swap version that did not exist before step 1
   (fragment ids are stable and new writes always create fragments), so the
   swap version's new fragment ids are recorded in the table's config under
   ``cognee.payload_migration`` and their rows get ``payload`` rebuilt from
   ``__payload_old`` with the same transform, merged by ``id``. A row another
   process updates after the swap is its writer's: the merge runs without
   Lance's own retry, so Lance refuses it over such a row, and the window
   fragments are scanned again, where the row no longer is. Compaction, the
   one thing that gives fragments new ids, leaves a table carrying one of the
   temporary columns alone; should the window fragments have been rewritten
   anyway, a row written in the window still has no ``payload``, so a null
   ``payload`` left after the repair refuses the drop and keeps the old column.
4. ``drop_columns`` removes ``__payload_old``; the record is deleted.

Each step is a committed Lance version and ``payload`` exists at every one of
them, so a reader never sees a table without it and an interrupted migration
leaves the table readable. The temporary columns are nullable (the swap also
makes the old column nullable), so a write that carries only ``id``,
``vector`` and ``payload`` still lands between the steps. The next run
finishes it: a leftover ``__payload_migrating`` is dropped and computed again;
a leftover ``__payload_old`` has its recorded window rows repaired and is
dropped. A table interrupted between the swap and its record is refused, with
the old column kept, since the window rows cannot be told from later writes.
The vector column's data files are never rewritten, and no row is ever
materialized as Python objects.

Nothing partial: a required target field that is missing from the stored
struct with no default, a value that cannot be cast to its new type, or a null
in a required field that no default can fill, fails the UDF, and
``add_columns`` commits nothing.
"""

from __future__ import annotations

import json
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
ID_COLUMN = "id"
#: Table config key holding the swap's new fragment ids until the old column is dropped.
STATE_KEY = "cognee.payload_migration"

_NO_DEFAULT = object()


_ABORTED = "LanceDB payload schema migration aborted"
# Lance's refusal of a merge over a row updated since it was planned, with its
# retry off (``conflict_retries(0)``).
_CONCURRENT_WRITER = "concurrent writers"


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


def _fragment_ids(dataset) -> set[int]:
    return {fragment.fragment_id for fragment in dataset.get_fragments()}


def _repair_window_rows(dataset, fragment_ids, stored, target, steps, batch_size) -> int:
    """Rebuild ``payload`` from ``__payload_old`` for the rows of ``fragment_ids``, the
    fragments written between the add-column commit and the swap, merged by id.
    Returns the number of rows repaired.

    Each batch is merged with Lance's own retry off: a retry would replay the
    batch over a row another process updated since the scan, whose new payload
    is its writer's. Lance refuses the commit instead, and the window fragments
    are scanned again at the latest version; an update moves its row to a new
    fragment, so the row is no longer there and every pass has fewer rows left.
    """
    repaired = 0
    while True:
        fragments = [f for f in dataset.get_fragments() if f.fragment_id in fragment_ids]
        if not fragments:
            return repaired
        scanner = dataset.scanner(
            fragments=fragments, columns=[ID_COLUMN, OLD_PAYLOAD_COLUMN], batch_size=batch_size
        )
        try:
            for batch in scanner.to_batches():
                if batch.num_rows == 0:
                    continue
                struct = _build_struct(batch.column(OLD_PAYLOAD_COLUMN), stored, target, steps)
                source = pa.table({ID_COLUMN: batch.column(ID_COLUMN), PAYLOAD_COLUMN: struct})
                (
                    dataset.merge_insert(ID_COLUMN)
                    .when_matched_update_all()
                    .conflict_retries(0)
                    .execute(source)
                )
                repaired += batch.num_rows
        except OSError as error:
            if _CONCURRENT_WRITER not in str(error):
                raise
            dataset.checkout_latest()
            continue
        return repaired


def _refuse_unrepaired_rows(dataset) -> None:
    """Keep ``__payload_old`` while a row has no ``payload``: its payload is there.

    Every row of the swap version has a ``payload`` (the UDF computed one) except
    the rows appended in the window, and every later write carries one, so after
    the repair none should be null. One that is sits in a window fragment the
    repair did not find: compaction gave the fragment a new id.
    """
    unrepaired = dataset.count_rows(filter=f"{PAYLOAD_COLUMN} IS NULL")
    if unrepaired:
        raise SchemaEvolutionError(
            f"{unrepaired} row(s) written during the migration are not in its recorded "
            f"window fragments (a compaction rewrote them) and hold their payload only in "
            f"'{OLD_PAYLOAD_COLUMN}', which is kept. No data was modified. Restore the rows "
            f"whose '{PAYLOAD_COLUMN}' is null from the old column by id, then drop it, and "
            "re-run migration."
        )


def _finish_interrupted(dataset, defaults, required, batch_size) -> int:
    """Bring a table an earlier migration left mid-way back to the start state and
    return the number of window rows repaired on the way.

    ``payload`` is present at every committed version, so only two states exist:
    before the swap (``__payload_migrating`` next to ``payload``: drop it, it is
    computed again against the current target) or after it (``__payload_old``
    next to the new ``payload``: repair the recorded window rows, drop it).
    """
    names = dataset.schema.names
    if PAYLOAD_COLUMN not in names:
        raise SchemaEvolutionError(
            f"the table has no '{PAYLOAD_COLUMN}' column (columns: {', '.join(names)}). "
            "No data was modified."
        )
    if MIGRATING_COLUMN in names:
        dataset.drop_columns([MIGRATING_COLUMN])
    repaired = 0
    if OLD_PAYLOAD_COLUMN in names:
        record = dataset.config().get(STATE_KEY)
        if record is None:
            raise SchemaEvolutionError(
                "an earlier run was interrupted between its swap and the record of the rows "
                f"written meanwhile, so '{OLD_PAYLOAD_COLUMN}' is kept: a row written then "
                f"holds its payload there and null or an outdated value in '{PAYLOAD_COLUMN}'. "
                "No data was modified. Restore such rows from the old column by id, then drop "
                "it, and re-run migration."
            )
        stored = dataset.schema.field(OLD_PAYLOAD_COLUMN).type
        current = dataset.schema.field(PAYLOAD_COLUMN).type
        steps = plan_payload_evolution(stored, current, defaults, required)
        window = set(json.loads(record)["window"])
        repaired = _repair_window_rows(dataset, window, stored, current, steps, batch_size)
        _refuse_unrepaired_rows(dataset)
        dataset.drop_columns([OLD_PAYLOAD_COLUMN])
    if STATE_KEY in dataset.config():
        dataset.delete_config_keys([STATE_KEY])
    return repaired


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
    repaired = _finish_interrupted(dataset, defaults, required, batch_size)

    stored = dataset.schema.field(PAYLOAD_COLUMN).type
    if not pa.types.is_struct(stored):
        raise SchemaEvolutionError(
            f"'{PAYLOAD_COLUMN}' is {stored}, not a struct. No data was modified."
        )
    if stored == target:
        # A migration interrupted after its swap: the payload is already the
        # target, only the old column was left to finish, done above.
        return _stats(dataset, [], stored, target, repaired)
    steps = plan_payload_evolution(stored, target, defaults, required)

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

    before = _fragment_ids(dataset)
    try:
        dataset.add_columns(transform, read_columns=[PAYLOAD_COLUMN], batch_size=batch_size)
    except Exception as error:
        # Lance reports an exception raised inside the UDF as an OSError carrying
        # its text; nothing was committed. Surface ours under its own type.
        if _ABORTED in str(error):
            raise SchemaEvolutionError(str(error).split(_ABORTED, 1)[1].strip(" :,\n")) from error
        if "commit conflict" in str(error).lower():
            raise SchemaEvolutionError(
                "another process wrote to the table while its new payload column was being "
                "computed, so Lance refused the commit. No data was modified. Re-run migration."
            ) from error
        raise
    # One commit: the table goes from the old ``payload`` to the new one with
    # no version in between that lacks the column. The old column turns
    # nullable in the same commit, for the writes that land before its drop.
    dataset.alter_columns(
        {"path": PAYLOAD_COLUMN, "name": OLD_PAYLOAD_COLUMN, "nullable": True},
        {"path": MIGRATING_COLUMN, "name": PAYLOAD_COLUMN},
    )
    # The handle is at the swap version: its fragments beyond ``before`` were
    # written in the window. Recorded first, so an interruption from here on
    # can finish exactly.
    window = sorted(_fragment_ids(dataset) - before)
    dataset.update_config({STATE_KEY: json.dumps({"window": window})})
    repaired += _repair_window_rows(dataset, set(window), stored, target, steps, batch_size)
    _refuse_unrepaired_rows(dataset)
    dataset.drop_columns([OLD_PAYLOAD_COLUMN])
    dataset.delete_config_keys([STATE_KEY])

    return _stats(dataset, steps, stored, target, repaired)


def _stats(dataset, steps, stored, target, repaired) -> dict[str, Any]:
    target_names = {field.name for field in target}
    return {
        "rows": dataset.count_rows(),
        "added": [step.name for step in steps if step.source in ("default", "null")],
        "removed": [
            stored.field(i).name
            for i in range(stored.num_fields)
            if stored.field(i).name not in target_names
        ],
        "cast": [step.name for step in steps if step.source == "cast"],
        "repaired": repaired,
    }
