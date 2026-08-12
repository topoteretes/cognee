"""Recall coverage: replay what an agent asked and report where memory fell short.

A run takes an ``agent_label`` (a tool such as Claude Code or Codex), collects the
recall questions that label produced in the recent window, judges whether memory
could answer them, and reports the gaps per topic, dataset and user.

Deliberately import-light: this package's ``__init__`` pulls in nothing, so
``import cognee.modules.recall_coverage.models`` (done for SQLAlchemy metadata
registration during relational-engine setup) cannot drag the LLM, embedding or
search stacks into an early import cycle. Import the submodule you need.
"""
