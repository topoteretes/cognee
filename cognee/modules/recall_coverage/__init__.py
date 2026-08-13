"""Recall coverage: replay what an agent asked and report where memory fell short.

A run takes an ``agent_label`` (a tool such as Claude Code or Codex, or ``all``),
collects the recall questions that label produced in the recent window, judges
whether memory could answer them, and reports one flat question table plus a
per-topic breakdown. Attribution — which agent, user or dataset a question came
from — is a column on each row, so the UI filters rather than the report
branching.

Deliberately import-light: this package's ``__init__`` pulls in nothing, so
``import cognee.modules.recall_coverage.models`` (done for SQLAlchemy metadata
registration during relational-engine setup) cannot drag the LLM, embedding or
search stacks into an early import cycle. Import the submodule you need.
"""
