"""Pre-assembled enrichment pipelines for ``improve()`` stages.

Each module builds one ready-to-run task list from the tasks in
``cognee/tasks/memify`` (feedback weights, entity consolidation,
session/agent-trace persistence, fact-conflict review, the global context
index) or ``cognee/tasks/code_docs`` (a code graph's docstrings as linked
chunks, ``cognify_code_docs``) and runs it through the internal
``cognee.modules.memify.run_memify_tasks`` runner.
"""
