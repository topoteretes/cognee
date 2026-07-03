# Changelog

All notable changes to the Cognee VS Code extension are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - Unreleased

### Added
- Editor-agnostic core (`src/core`) with a typed Cognee HTTP client, deterministic
  per-repository dataset scoping, configuration validation, and Evidence/citation parsing.
- Commands: **Ask My Project Memory**, **Recall**, **Remember Selection**, **Remember File**,
  **Index Workspace**, **Forget Project Memory**, and **Set Up**.
- "Ask my project memory" webview panel that renders answers with clickable citations to source files.
- Direct citation resolution via a per-workspace path index: files you remember are recorded with
  their exact relative path, so a citation resolves straight to the file you ingested — even when the
  workspace has several files of that name. Falls back to snippet-content matching, then a user pick.
- Unit tests running against a mocked Cognee backend (no live keys, no LLM in CI).
