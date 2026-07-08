"""The cognee_cli shim's static command table must match the real parser.

The previous fast-help attempt (minimal_cli.py) drifted to 5 of 21 commands
because nothing enforced sync. These tests are that enforcement: if a command
is added or removed in _discover_commands without updating the shim's help
screen, this fails in CI.
"""

import cognee_cli
from cognee.cli._cognee import _discover_commands


def _shim_commands() -> set:
    return {name for _group, commands in cognee_cli.COMMAND_GROUPS for name, _desc in commands}


def _real_commands() -> set:
    names = set()
    for command_class in _discover_commands():
        names.add(command_class.command_string)
    return names


def test_shim_help_lists_every_real_command():
    missing = _real_commands() - _shim_commands()
    assert not missing, (
        f"commands missing from cognee_cli.COMMAND_GROUPS (the static --help screen): {missing}"
    )


def test_shim_help_lists_no_phantom_commands():
    phantom = _shim_commands() - _real_commands()
    assert not phantom, (
        f"cognee_cli.COMMAND_GROUPS lists commands the real parser doesn't have: {phantom}"
    )


def test_cli_mode_set_for_commands_but_not_ui_launcher():
    """CLI mode (console log silencing) applies to one-shot commands only.
    `-ui` launches long-running servers whose startup logs must stay visible,
    and whose spawned backend would inherit the variable via os.environ."""
    assert cognee_cli._should_set_cli_mode(["add", "hello"]) is True
    assert cognee_cli._should_set_cli_mode(["cognify", "--verbose"]) is True
    assert cognee_cli._should_set_cli_mode(["doctor"]) is True
    assert cognee_cli._should_set_cli_mode(["-ui"]) is False
    assert cognee_cli._should_set_cli_mode(["--debug", "-ui"]) is False


def test_shim_fast_paths_do_not_import_cognee():
    """The whole point of the shim: help/version render without the ~1.5s
    cognee import (and its side effects). Run in a subprocess so an already
    imported cognee in this test process can't mask a regression."""
    import subprocess
    import sys

    code = (
        "import sys; import cognee_cli; cognee_cli._print_help(); "
        "cognee_cli._print_welcome(); "
        "assert 'cognee' not in sys.modules, 'shim fast paths imported cognee'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
