"""Automatic capture controls and secret redaction. Explicit remember is independent.

A port of the claude-code/codex ``_capture_policy.py`` (openclaw has the same
rules in ``capture-policy.ts``); the shared cases live in
``integrations/conformance/capture_policy_cases.json``. The rules are the
reference module's, line for line. The one difference: settings are read from a
mapping instead of ``os.environ``, so a value in ``$HERMES_HOME/cognee.json``
(``capture_tools``, ``capture_deny_tools``, ``capture_deny_paths``,
``capture_redact``, ``capture_redact_patterns``, ``capture``) works like its
``COGNEE_CAPTURE_*`` variable. :func:`policy_from_config` builds that mapping.

* ``COGNEE_CAPTURE_TOOLS`` (allow, default ``*``) and ``COGNEE_CAPTURE_DENY_TOOLS``
  take tool names, fnmatch globs, or ``Tool(prefix:*)`` matched against the first
  word(s) of a shell command, case-insensitively for Hermes' lowercase tool names.
* A call whose path-like params, or whose shell command's words, name a secret
  file (``.env*``, ``*.pem``, ssh and cloud credentials, ...) is never captured.
* ``COGNEE_CAPTURE_REDACT`` (on by default) scrubs private keys, connection
  strings, tokens, credential assignments and vendor keys from what is stored.

A pattern that cannot be parsed fails closed: :class:`CapturePatternError` is
raised and the caller captures nothing.
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
import shlex
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

_FALSE = {"0", "false", "no", "off"}
_PATH_KEYS = {"file_path", "filepath", "path", "paths", "notebook_path", "filename"}
_DENY_PATHS = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "id_rsa*",
    "id_ed25519*",
    ".npmrc",
    ".netrc",
    "*/.aws/credentials",
    "*/.ssh/*",
    "*.p12",
    "*.pfx",
    "secrets.*",
    "credentials.*",
)
# The last word of a secret-bearing name. ``secret`` may carry a ``_key`` /
# ``_access_key`` suffix (``SECRET_KEY``, ``aws_secret_access_key``), and the
# AWS / npm names a plain suffix test misses are spelled out: ``access_key_id``,
# camelCase ``SessionToken`` / ``SecretAccessKey`` / ``_authToken``.
_SECRET_WORD = (
    r"secret(?:[_-]?(?:access[_-]?)?key)?|token|password|passwd|api[_-]?key"
    r"|access[_-]?key(?:[_-]?id)?|session[_-]?token|auth[_-]?token"
)
_SECRET_KEY = re.compile(rf"(?i)^(?:authorization|x-api-key|_auth|(?:.*[_-])?(?:{_SECRET_WORD}))$")
_RULES = (
    (
        "private-key",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)",
    ),
    ("connection", r"(?i)\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqps?)://[^\s\"'<>]+"),
    ("authorization", r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=-]+"),
    (
        "credential",
        rf"(?i)[\"']?\b(?:(?:[\w-]*[_-])?(?:{_SECRET_WORD}|x-api-key)|_auth)"
        r"[\"']?\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)",
    ),
    (
        "vendor-key",
        r"\b(?:sk-(?:proj-|ant-)?|gh[pousr]_|github_pat_|xox[baprs]-|whsec_)[A-Za-z0-9_-]{16,}"
        r"|\bnpm_[A-Za-z0-9]{36}\b|\bpypi-[A-Za-z0-9_-]{50,}",
    ),
    (
        "aws-key-id",
        r"\b(?:A3T[A-Z0-9]|AKIA|ASIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|APKA|ABIA|ACCA)[A-Z0-9]{16}\b",
    ),
    ("bcrypt", r"\$2[aby]\$\d{2}\$[./A-Za-z0-9]{53}"),
)
_COMPILED = [(kind, re.compile(pattern)) for kind, pattern in _RULES]
# ``Tool(prefix:*)`` borrows Claude Code's permission spelling, so the pattern a
# user already writes to allow a command also selects whether it is remembered.
_COMMAND_PATTERN = re.compile(r"^(?P<tool>[^()\s]+)\((?P<prefix>[^()]*):\*\)$")

#: Config key -> the environment variable it stands for.
CONFIG_KEYS = {
    "capture": "COGNEE_CAPTURE",
    "capture_tools": "COGNEE_CAPTURE_TOOLS",
    "capture_deny_tools": "COGNEE_CAPTURE_DENY_TOOLS",
    "capture_deny_paths": "COGNEE_CAPTURE_DENY_PATHS",
    "capture_redact": "COGNEE_CAPTURE_REDACT",
    "capture_redact_patterns": "COGNEE_CAPTURE_REDACT_PATTERNS",
}


class CapturePatternError(ValueError):
    """A capture allow/deny entry, deny path list or redaction pattern is not understood."""


def _env_value(value: Any) -> str:
    """A config value as the string its environment variable would hold."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return json.dumps([str(item) for item in value])
    return "" if value is None else str(value)


def policy_from_config(config: Mapping[str, Any]) -> dict[str, str]:
    """The settings mapping for a loaded plugin config.

    ``load_config`` already falls back to each ``COGNEE_CAPTURE_*`` variable, so
    a key missing from ``config`` falls back to the environment here too.
    """
    settings: dict[str, str] = {}
    for key, env_name in CONFIG_KEYS.items():
        value = config.get(key)
        if value is None or value == "":
            value = os.environ.get(env_name, "")
        settings[env_name] = _env_value(value)
    return settings


def capture_enabled(env: Mapping[str, str]) -> bool:
    return (env.get("COGNEE_CAPTURE") or "true").strip().lower() not in _FALSE


def _list_env(env: Mapping[str, str], name: str, *, separator: str = ",") -> list[str]:
    raw = (env.get(name) or "").strip()
    if not raw:
        return []
    if raw.startswith("["):
        values = json.loads(raw)
        if not isinstance(values, list) or not all(isinstance(item, str) for item in values):
            raise ValueError(f"{name} must be an array of strings")
        return values
    return [part.strip() for part in raw.split(separator) if part.strip()]


def _glob_match(name: str, pattern: str) -> bool:
    try:
        return fnmatch.fnmatchcase(name, pattern)
    except re.error as exc:
        raise CapturePatternError(
            f"capture pattern {pattern!r} is not a valid glob: {exc}"
        ) from exc


def _sensitive_path(value, env: Mapping[str, str]) -> bool:
    if isinstance(value, (tuple, list)):
        return any(_sensitive_path(item, env) for item in value)
    if not isinstance(value, str):
        return False
    normalized = value.replace("\\", "/").lower()
    name = PurePosixPath(normalized).name
    return any(
        _glob_match(name, pattern.lower())
        or _glob_match("/" + normalized.lstrip("/"), pattern.lower())
        for pattern in (*_DENY_PATHS, *_list_env(env, "COGNEE_CAPTURE_DENY_PATHS"))
    )


# Shell control characters that end a word for the secret-file check:
# ``cat .env|grep``, ``cat<.env``, ``$(cat .env)``, ``a;cat .env``.
_SHELL_SPLIT = re.compile(r"[;|&()<>`]")
_HOME_VAR = re.compile(r"^\$(?:HOME|\{HOME\})(?=/|\Z)")


def _shell_words(command: str) -> list[str]:
    """Split a shell command into words, POSIX-quoted; whitespace if it will not parse."""
    try:
        words = shlex.split(command)
    except ValueError:  # unbalanced quote, trailing backslash
        words = command.split()
    pieces = []
    for word in words:
        for piece in _SHELL_SPLIT.split(word):
            if piece:
                pieces.append(_HOME_VAR.sub("~", piece))
    return pieces


def _command_reads_secret(params, env: Mapping[str, str]) -> bool:
    """Whether a shell command names a deny-listed path as one of its words.

    A word-level check, not a shell interpreter: ``cat ~/.aws/credentials``,
    ``less "$HOME/.ssh/id_rsa"`` and ``source .env.local`` are caught, but a
    path built at run time (``cat $AWS_DIR/credentials``) is not.
    """
    command = params.get("command") if isinstance(params, dict) else None
    if not isinstance(command, str):
        return False
    return any(_sensitive_path(word, env) for word in _shell_words(command))


def _has_sensitive_path(value, env: Mapping[str, str]) -> bool:
    if isinstance(value, dict):
        return any(
            (str(key).lower() in _PATH_KEYS and _sensitive_path(item, env))
            or _has_sensitive_path(item, env)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_has_sensitive_path(item, env) for item in value)
    return False


def _parse_pattern(pattern: str) -> tuple[str, str | None]:
    """Split ``Tool(prefix:*)`` into (tool glob, command prefix); plain names get None."""
    if "(" not in pattern and ")" not in pattern:
        return pattern, None
    match = _COMMAND_PATTERN.match(pattern)
    prefix = match.group("prefix").strip() if match else ""
    if not prefix:
        raise CapturePatternError(
            f"capture tool pattern {pattern!r} is not understood; "
            "use a tool name, an fnmatch glob, or Tool(command-prefix:*)"
        )
    return match.group("tool"), prefix


def _tool_patterns(
    env: Mapping[str, str], env_name: str, default: list[str]
) -> list[tuple[str, str | None]]:
    """Parse every entry up front so a bad one fails whether or not it is reached."""
    try:
        entries = _list_env(env, env_name, separator="|") or default
    except ValueError as exc:
        raise CapturePatternError(f"{env_name} is not a valid JSON array: {exc}") from exc
    try:
        return [_parse_pattern(pattern) for pattern in entries]
    except CapturePatternError as exc:
        raise CapturePatternError(f"{env_name}: {exc}") from exc


def _matches(tool: str, prefix: str | None, name: str, params) -> bool:
    # Case-insensitive, as in openclaw: ``Terminal`` and ``terminal`` are one tool.
    if not _glob_match(name.lower(), tool.lower()):
        return False
    if prefix is None:
        return True
    command = params.get("command") if isinstance(params, dict) else None
    if not isinstance(command, str):
        return False
    command = command.lstrip()
    if not command.startswith(prefix):
        return False
    return len(command) == len(prefix) or command[len(prefix)].isspace()


def allow_tool(name: str, params, env: Mapping[str, str]) -> bool:
    """Whether a tool call may be captured.

    Raises :class:`CapturePatternError` when an allow or deny entry cannot be
    parsed, so a misspelled pattern never silently disables capture.
    """
    if not capture_enabled(env):
        return False
    allowed = _tool_patterns(env, "COGNEE_CAPTURE_TOOLS", ["*"])
    denied = _tool_patterns(env, "COGNEE_CAPTURE_DENY_TOOLS", [])
    if not any(_matches(tool, prefix, name, params) for tool, prefix in allowed):
        return False
    if any(_matches(tool, prefix, name, params) for tool, prefix in denied):
        return False
    return not (_has_sensitive_path(params, env) or _command_reads_secret(params, env))


def capture_pattern_error(env: Mapping[str, str]) -> str:
    """The message for a malformed capture setting, or ''. Cheap enough per session."""
    try:
        _tool_patterns(env, "COGNEE_CAPTURE_TOOLS", ["*"])
        _tool_patterns(env, "COGNEE_CAPTURE_DENY_TOOLS", [])
    except CapturePatternError as exc:
        return str(exc)
    try:
        _list_env(env, "COGNEE_CAPTURE_DENY_PATHS")
    except ValueError as exc:
        return f"COGNEE_CAPTURE_DENY_PATHS is not a valid JSON array: {exc}"
    try:
        for pattern in _list_env(env, "COGNEE_CAPTURE_REDACT_PATTERNS", separator="\n"):
            re.compile(pattern)
    except (ValueError, re.error) as exc:
        return f"COGNEE_CAPTURE_REDACT_PATTERNS is not understood: {exc}"
    return ""


def redact(value, env: Mapping[str, str]):
    """Redact before truncation so a clipped private key cannot escape matching."""
    if (env.get("COGNEE_CAPTURE_REDACT") or "true").strip().lower() in _FALSE:
        return value
    if isinstance(value, str):
        for kind, pattern in _COMPILED:
            value = pattern.sub(f"[redacted:{kind}]", value)
        # User-supplied regexes are JSON strings; commas are valid regex syntax.
        raw = (env.get("COGNEE_CAPTURE_REDACT_PATTERNS") or "").strip()
        if raw:
            patterns = _list_env(env, "COGNEE_CAPTURE_REDACT_PATTERNS", separator="\n")
            for pattern in patterns:
                value = re.sub(pattern, "[redacted:custom]", value)
        return value
    if isinstance(value, dict):
        return {
            redact(key, env): "[redacted:credential]"
            if _SECRET_KEY.match(str(key))
            else redact(item, env)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item, env) for item in value]
    return value


def is_self_reference(name: str, params) -> bool:
    """A call that is the plugin talking to itself, and would recurse if stored.

    The cognee tools this provider registers, and (as in the claude-code, codex
    and openclaw plugins) a shell command that mentions cognee.
    """
    if str(name or "").lower().startswith("cognee_"):
        return True
    command = params.get("command") if isinstance(params, dict) else None
    return isinstance(command, str) and "cognee" in command
