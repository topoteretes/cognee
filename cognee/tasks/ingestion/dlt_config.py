"""Per-call options for DLT (relational) ingestion, grouped as ``dlt_config``.

``add(..., dlt_config={...})`` — and ``remember``, which forwards it to add —
is the one place a caller sets options that apply only to dlt sources:
connection strings, dlt resources and CSV files. The same options are still
accepted as bare keyword arguments (``add(..., primary_key="id")``) for
compatibility; an option given both ways is an error, never a merge.

The read options (``primary_key``, ``write_disposition``, ``query``,
``max_rows_per_table``) decide what dlt reads from the source and apply to
every dlt source, connectors included. The two selections
(``column_value_columns``, ``temporal_columns``) shape the graph built from
relational rows and are ignored by document-tagged connector sources. A key
left unset falls back to its ``IngestionConfig`` (``DLT_*`` env) default
where the option is consumed.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict

from cognee.infrastructure.loaders.external.dlt_csv_loader import DltCsvLoader


class DltConfig(BaseModel):
    """The ``dlt_config`` dict, validated: an unknown key raises."""

    model_config = ConfigDict(extra="forbid")

    primary_key: str | None = None
    write_disposition: str | None = None
    query: str | None = None
    max_rows_per_table: int | None = None
    column_value_columns: dict[str, list[str]] | None = None
    temporal_columns: dict[str, list[str]] | None = None


DLT_OPTION_NAMES = frozenset(DltConfig.model_fields)


def resolve_dlt_options(dlt_config: dict | None, loose_kwargs: dict[str, Any]) -> dict[str, Any]:
    """The DLT options a call set, from ``dlt_config`` and from bare keyword arguments.

    Only keys the caller set are returned, so the consumers' defaults still
    apply. A key present in both spellings raises: silently preferring one
    would hide the mistake.
    """
    grouped = DltConfig(**(dlt_config or {})).model_dump(exclude_unset=True)
    loose = {key: value for key, value in loose_kwargs.items() if key in DLT_OPTION_NAMES}
    duplicated = sorted(set(grouped) & set(loose))
    if duplicated:
        raise ValueError(
            "DLT option(s) given both in dlt_config and as keyword argument(s): "
            + ", ".join(duplicated)
        )
    # Validate the loose values through the same model as the grouped ones.
    return DltConfig(**grouped, **loose).model_dump(exclude_unset=True)


def with_csv_loader_options(
    preferred_loaders: dict[str, dict[str, Any]] | None, dlt_options: dict[str, Any]
) -> dict[str, dict[str, Any]] | None:
    """``preferred_loaders`` with the DLT options folded into the CSV loader's entry.

    CSV files reach dlt through the loader engine, which hands a loader the
    options stored under its name in ``preferred_loaders``; this is how the
    options given to ``add()`` reach a CSV too. Naming the loader here does
    not change dispatch: it only handles CSV files and already outranks the
    plain ``csv_loader``. An option also set under the loader's entry raises.
    """
    if not dlt_options:
        return preferred_loaders
    loaders = dict(preferred_loaders or {})
    loader_options = dict(loaders.get(DltCsvLoader.loader_name, {}))
    duplicated = sorted(set(loader_options) & set(dlt_options))
    if duplicated:
        raise ValueError(
            f"DLT option(s) given both to add() and under preferred_loaders"
            f"[{DltCsvLoader.loader_name!r}]: " + ", ".join(duplicated)
        )
    loaders[DltCsvLoader.loader_name] = {**loader_options, **dlt_options}
    return loaders
