from sqlalchemy import JSON, Boolean, Column, Float, Integer, String, Text

from cognee.infrastructure.databases.relational import Base


class EvaluationScope(Base):
    __tablename__ = "answer_evaluation_scopes"
    id = Column(String(64), primary_key=True)
    owner_id = Column(String(36), nullable=False, index=True)
    dataset_id = Column(String(36), nullable=False)
    enabled = Column(Boolean, nullable=False, default=True)
    revision = Column(Integer, nullable=False, default=0)
    synced_revision = Column(Integer, nullable=False, default=-1)
    last_run_at = Column(Float, nullable=False, default=0)


class EvaluationTrace(Base):
    __tablename__ = "answer_evaluation_traces"
    id = Column(String(64), primary_key=True)
    scope_id = Column(String(64), nullable=False, index=True)
    fingerprint = Column(String(64), nullable=False)
    created_at = Column(Float, nullable=False)
    expires_at = Column(Float, nullable=False)
    priority = Column(Float, nullable=False)
    # Null after eviction/deletion: bounded receipt prevents replay resurrection.
    ciphertext = Column(Text, nullable=True)


class EvaluationRun(Base):
    __tablename__ = "answer_evaluation_runs"
    id = Column(String(36), primary_key=True)
    scope_id = Column(String(64), nullable=False, index=True)
    created_at = Column(Float, nullable=False)
    status = Column(String(24), nullable=False)
    # IDs/versions and numeric verdicts only. Never copies prompts/answers here.
    result = Column(JSON, nullable=False)
