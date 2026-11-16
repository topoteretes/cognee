"""Key pipeline_run_shard_items by an item's position in its shard

Revision ID: 9b4d7e2a6c1f
Revises: 3cb1c5e71f26
Create Date: 2026-11-16 23:00:00.000000

An item of a shard was always a row of ``data``, so its row was keyed by
``(shard_id, data_id)``. An item can now also be a value that is no row, so the
row is keyed by the item's position in its shard (``item_index``), ``data_id``
may be NULL, and the value is kept in ``value``.

On PostgreSQL the table is changed in place and every row keeps its place: its
position is its rank among its shard's rows by ``data_id``, the order a worker
read them in. On any other database no run hands shards out (workers need
PostgreSQL), so the table is empty and is created again in its new shape; a
table with rows there is refused, not emptied.

Idempotent: every step is inspector-guarded on its own, so an upgrade or
downgrade interrupted between two statements finishes on rerun.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9b4d7e2a6c1f"
down_revision: str | None = "3cb1c5e71f26"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SHARDS = "pipeline_run_shards"
ITEMS = "pipeline_run_shard_items"
DATA_ID_INDEX = "ix_pipeline_run_shard_items_data_id"
BY_POSITION = ["shard_id", "item_index"]
BY_DATA_ROW = ["shard_id", "data_id"]


def _inspector():
    """A fresh inspector per check: one created before a DDL statement does not see it."""
    return sa.inspect(op.get_bind())


def _columns() -> dict[str, dict]:
    return {column["name"]: column for column in _inspector().get_columns(ITEMS)}


def _create(by_position: bool) -> None:
    """The item table in its new shape, or in the one the previous revision made."""
    key = (
        [
            sa.Column("item_index", sa.Integer(), nullable=False),
            sa.Column("data_id", sa.Uuid(), nullable=True),
            sa.Column("value", sa.JSON(), nullable=True),
        ]
        if by_position
        else [sa.Column("data_id", sa.Uuid(), nullable=False)]
    )
    op.create_table(
        ITEMS,
        sa.Column("shard_id", sa.Uuid(), nullable=False),
        *key,
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("error_class", sa.String(), nullable=True),
        sa.Column("error_message", sa.String(), nullable=True),
        sa.ForeignKeyConstraint(["shard_id"], [f"{SHARDS}.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["data_id"], ["data.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint(*(BY_POSITION if by_position else BY_DATA_ROW)),
    )


def _create_again(by_position: bool) -> None:
    """Where the table cannot be changed in place: it is empty, so make it anew."""
    if _inspector().has_table(ITEMS):
        if ("item_index" in _columns()) == by_position:
            return
        rows = op.get_bind().execute(sa.text(f"SELECT count(*) FROM {ITEMS}")).scalar()
        if rows:
            raise RuntimeError(
                f"{ITEMS} holds {rows} rows on {op.get_bind().dialect.name}, where no run "
                "hands shards out (workers need PostgreSQL), so this revision has no rule "
                "for keeping them. Remove them, or move the database to PostgreSQL first."
            )
        op.drop_table(ITEMS)
    _create(by_position)
    if DATA_ID_INDEX not in {index["name"] for index in _inspector().get_indexes(ITEMS)}:
        op.create_index(DATA_ID_INDEX, ITEMS, ["data_id"], unique=False)


def _set_primary_key(columns: list[str]) -> None:
    key = _inspector().get_pk_constraint(ITEMS)
    if key["constrained_columns"] == columns:
        return
    if key["constrained_columns"]:
        op.drop_constraint(key["name"], ITEMS, type_="primary")
    op.create_primary_key(f"{ITEMS}_pkey", ITEMS, columns)


def upgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        _create_again(by_position=True)
        return

    if "item_index" not in _columns():
        op.add_column(ITEMS, sa.Column("item_index", sa.Integer(), nullable=True))
    # A row's position is its rank in its shard by data_id, from 0: the order
    # the shard's worker read its items in.
    op.execute(
        f"""
        UPDATE {ITEMS} AS item SET item_index = ranked.position
        FROM (
            SELECT shard_id, data_id,
                   row_number() OVER (PARTITION BY shard_id ORDER BY data_id) - 1 AS position
            FROM {ITEMS} WHERE item_index IS NULL
        ) AS ranked
        WHERE item.shard_id = ranked.shard_id AND item.data_id = ranked.data_id
        """
    )
    if _columns()["item_index"]["nullable"]:
        op.alter_column(ITEMS, "item_index", nullable=False)
    if "value" not in _columns():
        op.add_column(ITEMS, sa.Column("value", sa.JSON(), nullable=True))
    _set_primary_key(BY_POSITION)
    if not _columns()["data_id"]["nullable"]:
        op.alter_column(ITEMS, "data_id", nullable=True)


def downgrade() -> None:
    if op.get_bind().dialect.name != "postgresql":
        _create_again(by_position=False)
        return

    if "item_index" not in _columns():
        return
    # The earlier shape keys a row by its data row: an item that is a value has
    # none, so its row cannot be kept.
    op.execute(f"DELETE FROM {ITEMS} WHERE data_id IS NULL")
    if _columns()["data_id"]["nullable"]:
        op.alter_column(ITEMS, "data_id", nullable=False)
    _set_primary_key(BY_DATA_ROW)
    if "value" in _columns():
        op.drop_column(ITEMS, "value")
    op.drop_column(ITEMS, "item_index")
