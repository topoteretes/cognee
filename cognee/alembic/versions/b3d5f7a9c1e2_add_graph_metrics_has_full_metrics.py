"""add_graph_metrics_has_full_metrics

Revision ID: b3d5f7a9c1e2
Revises: c4e8a1f6b3d7
Create Date: 2026-08-21 00:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b3d5f7a9c1e2"
down_revision: Union[str, None] = "c4e8a1f6b3d7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _get_column(inspector, table, name, schema=None):
    for col in inspector.get_columns(table, schema=schema):
        if col["name"] == name:
            return col
    return None


def upgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    if "graph_metrics" not in insp.get_table_names():
        return

    if _get_column(insp, "graph_metrics", "has_full_metrics"):
        return

    op.add_column(
        "graph_metrics",
        sa.Column(
            "has_full_metrics",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
    )

    # Existing rows predate the flag, so it has to be inferred. `diameter` is
    # the discriminator, not `num_tokens`: every graph-metrics adapter always
    # sets `diameter` on the full-computation path (a real value or a -1
    # placeholder, never NULL), while the node/edge counting path never
    # touches it. `num_tokens` can't be used for this: it's a SQL SUM() that
    # legitimately comes back NULL for a fully-computed run over a dataset
    # with no token_count data, which would misclassify that row as partial.
    # Rows that stay False are exactly the partial ones, which the metrics
    # endpoint will now recompute once instead of serving forever as complete.
    op.execute(
        sa.text("UPDATE graph_metrics SET has_full_metrics = true WHERE diameter IS NOT NULL")
    )


def downgrade() -> None:
    conn = op.get_bind()
    insp = sa.inspect(conn)

    if "graph_metrics" not in insp.get_table_names():
        return

    if _get_column(insp, "graph_metrics", "has_full_metrics"):
        op.drop_column("graph_metrics", "has_full_metrics")
