"""Schema guards for the five recall-coverage tables and their migration.

Three invariants this file exists to protect:

* **Statuses are ``Column(String)``, never ``Enum(...)``.** A native Postgres
  enum needs raw DDL to gain one value (see
  ``cognee/alembic/versions/1d0bb7fede17_add_pipeline_run_status.py``), so the
  enums live in the app layer (``recall_coverage.types``).
* **Owner/user/dataset ids are bare indexed UUIDs with no foreign keys**, like
  ``queries.user_id`` — and there is no ``agent_id`` column anywhere, nor a
  ``recall_coverage_question_datasets`` table.
* **The migration and the models cannot drift.** The revision is executed
  against a real (in-memory SQLite) database and the resulting columns and
  indexes are compared to ``Base.metadata``.
"""

import importlib.util
from pathlib import Path
from uuid import uuid4

import pytest
import sqlalchemy as sa
from alembic.migration import MigrationContext
from alembic.operations import Operations
from alembic.script import ScriptDirectory
from sqlalchemy.orm import Session

from cognee.infrastructure.databases.relational.ModelBase import Base
from cognee.modules.recall_coverage.models import (
    RecallCoverageCuratedQuestion,
    RecallCoverageQuestion,
    RecallCoverageRun,
    RecallCoverageTopic,
    RecallCoverageTopicSuggestion,
)

EXPECTED_TABLES = {
    "recall_coverage_runs",
    "recall_coverage_questions",
    "recall_coverage_topics",
    "recall_coverage_topic_suggestions",
    "recall_coverage_curated_questions",
}

MODELS = (
    RecallCoverageRun,
    RecallCoverageQuestion,
    RecallCoverageTopic,
    RecallCoverageTopicSuggestion,
    RecallCoverageCuratedQuestion,
)

ALEMBIC_DIR = Path(__file__).resolve().parents[4] / "alembic"
REVISION_PATH = ALEMBIC_DIR / "versions" / "c7e9a1b3d5f2_add_recall_coverage_tables.py"

# The head this revision was written against.
PREVIOUS_HEAD = "a7c3e9f1b5d8"
REVISION_ID = "c7e9a1b3d5f2"


def _revision_module():
    spec = importlib.util.spec_from_file_location("_recall_coverage_revision", REVISION_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sqlite_engine():
    return sa.create_engine("sqlite://")


def test_exactly_five_tables_are_registered():
    """The models register five tables and nothing else."""
    registered = {name for name in Base.metadata.tables if name.startswith("recall_coverage")}
    assert registered == EXPECTED_TABLES

    # A question row is (user_id, dataset_id, canonical text); there are no
    # per-dataset child cell rows.
    assert "recall_coverage_question_datasets" not in Base.metadata.tables


@pytest.mark.parametrize(
    "table_name,column_name",
    [
        ("recall_coverage_runs", "status"),
        ("recall_coverage_topic_suggestions", "status"),
        ("recall_coverage_curated_questions", "scope"),
        ("recall_coverage_questions", "source"),
    ],
)
def test_status_like_columns_are_plain_strings(table_name, column_name):
    """No native enum types: adding a value must stay an application change."""
    column = Base.metadata.tables[table_name].columns[column_name]
    assert isinstance(column.type, sa.String)
    assert not isinstance(column.type, sa.Enum)


@pytest.mark.parametrize("model", MODELS, ids=[model.__tablename__ for model in MODELS])
def test_no_foreign_keys_and_no_agent_id_column(model):
    """Owner/user/dataset ids are opaque UUIDs; the agent is a label, not an id."""
    table = model.__table__
    assert table.foreign_keys == set()
    assert "agent_id" not in table.columns
    assert "merged_into_id" not in table.columns
    assert "best_dataset_id" not in table.columns


@pytest.mark.parametrize(
    "table_name,column_name",
    [
        ("recall_coverage_runs", "owner_id"),
        ("recall_coverage_runs", "agent_label"),
        ("recall_coverage_questions", "run_id"),
        ("recall_coverage_questions", "user_id"),
        ("recall_coverage_questions", "dataset_id"),
        ("recall_coverage_questions", "question_group_id"),
        ("recall_coverage_topics", "owner_id"),
        ("recall_coverage_topic_suggestions", "owner_id"),
        ("recall_coverage_topic_suggestions", "status"),
        ("recall_coverage_curated_questions", "owner_id"),
    ],
)
def test_lookup_columns_are_indexed(table_name, column_name):
    table = Base.metadata.tables[table_name]
    indexed = {
        column.name
        for index in table.indexes
        for column in index.columns
        if len(index.columns) == 1
    }
    assert column_name in indexed


def test_composite_indexes_exist():
    """Both composite indexes the read paths depend on are declared."""
    topic_indexes = {index.name for index in RecallCoverageTopic.__table__.indexes}
    assert "ix_recall_coverage_topics_owner_deleted" in topic_indexes

    curated_indexes = {index.name for index in RecallCoverageCuratedQuestion.__table__.indexes}
    assert "ix_recall_coverage_curated_questions_owner_scope" in curated_indexes


def test_dataset_id_is_nullable_and_user_id_is_not():
    """A question with no single dataset is a real row, not a missing one."""
    questions = RecallCoverageQuestion.__table__
    assert questions.columns["dataset_id"].nullable is True
    assert questions.columns["user_id"].nullable is False
    # NULL topic_id is the sink, which is the wire literal "other", not a row.
    assert questions.columns["topic_id"].nullable is True


def test_run_row_defaults_round_trip():
    """A freshly inserted run is pending with zeroed counters and both timestamps."""
    engine = _sqlite_engine()
    Base.metadata.create_all(engine, tables=[model.__table__ for model in MODELS])

    run_id = uuid4()
    with Session(engine) as session:
        session.add(RecallCoverageRun(id=run_id, agent_label="all", owner_id=uuid4()))
        session.commit()

    with Session(engine) as session:
        run = session.get(RecallCoverageRun, run_id)
        assert run.status == "pending"
        assert run.recall_row_count == 0
        assert run.collapsed_retry_count == 0
        assert run.taxonomy_version == 0
        assert run.finished_at is None
        assert run.created_at is not None
        assert run.updated_at is not None

    engine.dispose()


def test_question_row_defaults_round_trip():
    """Scores stay NULL until the judge writes them; observed is the default source."""
    engine = _sqlite_engine()
    Base.metadata.create_all(engine, tables=[model.__table__ for model in MODELS])

    question_id = uuid4()
    with Session(engine) as session:
        session.add(
            RecallCoverageQuestion(
                id=question_id,
                run_id=uuid4(),
                user_id=uuid4(),
                dataset_id=None,
                question_text="What is our incident escalation path out of hours?",
            )
        )
        session.commit()

    with Session(engine) as session:
        question = session.get(RecallCoverageQuestion, question_id)
        assert question.source == "observed"
        assert question.was_asked is True
        assert question.occurrence_count == 0
        assert question.judge_score is None
        assert question.judge_answered is None
        assert question.impact is None
        assert question.dataset_id is None

    engine.dispose()


def test_revision_chains_onto_the_previous_head():
    module = _revision_module()
    assert module.revision == REVISION_ID
    assert module.down_revision == PREVIOUS_HEAD


def test_migration_tree_has_a_single_head():
    """Adding this revision must not fork the migration graph."""
    script_directory = ScriptDirectory(str(ALEMBIC_DIR))
    assert list(script_directory.get_heads()) == [REVISION_ID]


def test_migration_creates_exactly_what_the_models_declare():
    """Guard against model/migration drift in columns, nullability and indexes."""
    module = _revision_module()

    engine = _sqlite_engine()
    with engine.begin() as connection:
        with Operations.context(MigrationContext.configure(connection)):
            module.upgrade()

        inspector = sa.inspect(connection)
        assert set(inspector.get_table_names()) == EXPECTED_TABLES

        for table_name in sorted(EXPECTED_TABLES):
            model_table = Base.metadata.tables[table_name]

            migrated_columns = {
                column["name"]: column for column in inspector.get_columns(table_name)
            }
            assert set(migrated_columns) == set(model_table.columns.keys()), table_name

            for name, column in model_table.columns.items():
                # SQLite reports the primary key as NOT NULL regardless.
                if column.primary_key:
                    continue
                assert migrated_columns[name]["nullable"] == column.nullable, (
                    f"{table_name}.{name} nullability differs between model and migration"
                )

            migrated_indexes = {index["name"] for index in inspector.get_indexes(table_name)}
            assert migrated_indexes == {index.name for index in model_table.indexes}, table_name

    engine.dispose()


def test_migration_is_idempotent_when_create_all_already_ran():
    """Base.metadata.create_all() can beat alembic to it on a fresh database."""
    module = _revision_module()

    engine = _sqlite_engine()
    Base.metadata.create_all(engine, tables=[model.__table__ for model in MODELS])

    with engine.begin() as connection:
        with Operations.context(MigrationContext.configure(connection)):
            module.upgrade()

        assert set(sa.inspect(connection).get_table_names()) == EXPECTED_TABLES

    with engine.begin() as connection:
        with Operations.context(MigrationContext.configure(connection)):
            module.downgrade()

        assert not set(sa.inspect(connection).get_table_names()) & EXPECTED_TABLES

    engine.dispose()
