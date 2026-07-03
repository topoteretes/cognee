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
- Snippet-aware citation resolution: when several files share the cited basename, the one whose
  content contains the snippet is opened; only otherwise is the user asked to pick.
- Unit tests running against a mocked Cognee backend (no live keys, no LLM in CI).
