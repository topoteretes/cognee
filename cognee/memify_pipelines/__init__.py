"""Pre-assembled enrichment pipelines for ``improve()`` / ``memify()``.

Each module builds one ready-to-run task list from the tasks in
``cognee/tasks/memify`` (triplet embeddings, feedback and frequency weights,
entity consolidation, session/agent-trace persistence, the global context
index). ``memify_default_tasks.py`` is what ``improve()`` runs with no
options; ``memify_task_registry.py`` maps the string names the HTTP API
accepts onto these pipelines.
"""
