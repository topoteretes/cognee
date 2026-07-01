"""First-party integrations built on top of the cognee memory API.

Sub-packages here are thin, framework-agnostic layers that map an external
system onto cognee's ``remember`` / ``recall`` / ``forget`` primitives. They
never reach past those primitives into cognee internals, so they stay small
and stable across cognee releases.

Currently ships:

* :mod:`cognee.integrations.chat_memory` — a shared "chat memory adapter"
  core that every cognee-powered chat bot (Slack, Telegram, Discord, …)
  plugs into, so each bot stays thin (~100 lines) and they all share one
  consistent memory model.
"""
