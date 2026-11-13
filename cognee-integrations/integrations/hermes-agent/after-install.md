# Cognee Hermes Memory Plugin

The Cognee memory plugin has been installed. It does nothing until it is your
memory provider, so finish with one command in a terminal:

```bash
hermes memory setup cognee
```

This makes Cognee the memory provider and asks for a mode:

- **remote** (the default) — Cognee Cloud or a self-hosted cognee server. You'll
  be asked for the service URL and an API key (for Cognee Cloud, both come from
  https://platform.cognee.ai/). Nothing else to install.
- **local** — the plugin runs a cognee server on your machine (shared with the
  Claude Code / Codex / OpenClaw cognee plugins, if you use them). You'll be
  asked for an LLM API key, which cognee uses to build the knowledge graph
  (leave it blank to use local models instead), and offered to install the
  cognee runtime (`cognee==1.6.6`, about 1 GB from
  PyPI) into a private environment kept apart from Hermes's own. You can also
  install it later with `hermes cognee install-runtime`.

Then start a **new chat** in Hermes Desktop (or a new `hermes` session): a chat
that is already open keeps the memory it started with. Check with
`hermes memory status` (`Provider: cognee`) and `hermes cognee status`, then ask
the agent to remember something and, in another new chat, to recall it.

`hermes plugins enable cognee` and the Desktop toggle enable the plugin but do
not make it the memory provider; `hermes memory setup cognee` does.

## Updates

For a Hermes catalog installation, run `hermes plugins update cognee` to get
the latest reviewed commit. Do not overwrite it with a pip or checkout copy.

For a pip installation, run `pip install -U cognee-integration-hermes-agent`
followed by `cognee-hermes-install` to refresh the directory copy. The pip
installer refuses to overwrite a catalog-managed installation.

In local mode, an update that moves the pinned cognee turns local mode off until
you run `hermes cognee install-runtime`; Hermes says so when it reports Cognee as
unavailable. The command says which local cognee server this plugin started it
will stop and asks first (`--yes` without a terminal; other cognee plugins
sharing it start a new one on their next request), upgrades the runtime in
place, and then you start a new Hermes session. If a cognee server older than
the pinned cognee (or reporting no version) that it cannot stop still answers on
the local port, it changes nothing and names the server to stop before you
re-run it.

A dataset other than the default that you configured keeps recall on that
dataset while no recall mode is set, as before; the first reply of a session
says once that Cognee can also pick the dataset for each prompt
(`recall_mode: auto` in cognee.json or `COGNEE_PLUGIN_RECALL_MODE=auto`).
