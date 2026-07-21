from __future__ import annotations

import argparse
import json
import subprocess

import cognee.cli.echo as fmt
from cognee.cli import DEFAULT_DOCS_URL
from cognee.cli.commands.installers import REGISTRY, get_installer
from cognee.cli.reference import SupportsCliCommand


class InstallCommand(SupportsCliCommand):
    command_string = "install"
    help_string = (
        "Configure a coding agent (Claude Code, Cursor, OpenCode) to use cognee as its MCP memory"
    )
    docs_url = DEFAULT_DOCS_URL
    description = """
Configure a coding agent host to use cognee as its MCP memory server.

Writes the host's MCP config entry pointing at cognee-mcp, backs up the
original config, and probes the server to verify it works.

Supported hosts: claude-code, cursor, opencode

Examples:
  cognee install claude-code                  # install for Claude Code
  cognee install cursor                       # install for Cursor (user scope)
  cognee install cursor --scope project       # install at project level (.cursor/mcp.json)
  cognee install claude-code --scope project  # install at project level (.mcp.json)
  cognee install opencode                # install for OpenCode
  cognee install --list                  # show all supported hosts
  cognee install --dry-run cursor        # preview what would be written
  cognee install --uninstall cursor      # remove the cognee entry

When cognee-mcp is not on PATH, pass --mcp-dir:
  cognee install cursor --mcp-dir /path/to/cognee-mcp
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "harness",
            nargs="?",
            choices=sorted(REGISTRY),
            metavar="HARNESS",
            help=f"Coding agent to configure. One of: {', '.join(sorted(REGISTRY))}",
        )
        parser.add_argument(
            "--list",
            action="store_true",
            help="List all supported coding agents and exit",
        )
        parser.add_argument(
            "--uninstall",
            action="store_true",
            help="Remove the cognee MCP entry instead of adding it",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be written without actually writing anything",
        )
        parser.add_argument(
            "--mcp-dir",
            metavar="DIR",
            help=(
                "Path to the cognee-mcp source directory. "
                "Required when cognee-mcp is not on PATH. "
                "Produces: uv --directory DIR run cognee-mcp"
            ),
        )
        parser.add_argument(
            "--scope",
            choices=["user", "project"],
            default="user",
            help=(
                "Config scope. 'user' writes to the global config (default). "
                "'project' writes to the project-level config. "
                "Cursor writes to .cursor/mcp.json; Claude Code writes to .mcp.json. "
                "OpenCode ignores scope."
            ),
        )

    def execute(self, args: argparse.Namespace) -> None:
        if args.list:
            _print_harness_list()
            return

        if not args.harness:
            fmt.error(
                "Please specify a harness. "
                f"Supported: {', '.join(sorted(REGISTRY))}  "
                "Run `cognee install --list` to see all options."
            )
            return

        try:
            installer = get_installer(args.harness)
        except KeyError as exc:
            fmt.error(str(exc))
            return

        try:
            if args.uninstall:
                result = installer.uninstall(
                    scope=args.scope,
                    dry_run=args.dry_run,
                )
                fmt.success(result)
            else:
                result = installer.install(
                    mcp_dir=args.mcp_dir,
                    scope=args.scope,
                    dry_run=args.dry_run,
                )
                fmt.success(result)

                if not args.dry_run and "Already installed" not in result:
                    mcp_block = installer.build_mcp_block(args.mcp_dir)
                    _verify(mcp_block)

        except RuntimeError as exc:
            fmt.error(str(exc))
        except Exception as exc:  # noqa: BLE001
            fmt.error(f"Unexpected error: {exc}")


def _print_harness_list() -> None:
    fmt.note("Supported coding agents:")
    for name in sorted(REGISTRY):
        print(f"  {name}")


def _verify(mcp_block: dict) -> None:
    """Probe cognee-mcp over stdio to confirm the MCP handshake works.

    Sends:  initialize (JSON-RPC 2.0)  →  tools/list
    Timeout: 5 seconds.
    Always non-fatal — config has already been written.
    """
    command = mcp_block["command"]
    args = mcp_block.get("args", [])
    cmd = [command] + list(args)

    # Minimal MCP JSON-RPC messages (newline-delimited framing)
    init_msg = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "cognee-install-probe", "version": "1.0"},
            },
        }
    )
    list_msg = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }
    )
    stdin_payload = init_msg + "\n" + list_msg + "\n"

    try:
        result = subprocess.run(
            cmd,
            input=stdin_payload,
            capture_output=True,
            text=True,
            timeout=5,
        )
        stdout = result.stdout.strip()
        if stdout:
            for line in reversed(stdout.splitlines()):
                line = line.strip()
                if line:
                    try:
                        json.loads(line)
                        fmt.success("cognee-mcp responded to tools/list ✓")
                    except json.JSONDecodeError:
                        fmt.warning(
                            "cognee-mcp responded but output was not valid JSON — "
                            "config is written; verify manually."
                        )
                    return
        fmt.warning(
            "cognee-mcp produced no output during probe — "
            "config is written; verify manually by running: "
            f"{' '.join(cmd)}"
        )
    except subprocess.TimeoutExpired:
        fmt.warning(
            "cognee-mcp probe timed out (5s) — "
            "config is written and should work when the host launches it."
        )
    except FileNotFoundError:
        fmt.warning(
            f"Could not find {cmd[0]!r} for probe — "
            "config is written; install cognee-mcp to verify."
        )
    except Exception as exc:  # noqa: BLE001
        fmt.warning(f"Probe failed ({exc}) — config is written; verify manually.")
