import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Optional

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException


# MCP targets: write a config file
MCP_TARGETS = {"cursor", "claude", "vscode"}

# Tools targets: print integration instructions
TOOLS_TARGETS = {"openai", "langchain", "crewai", "anthropic"}

ALL_TARGETS = MCP_TARGETS | TOOLS_TARGETS


def _find_cognee_mcp_path() -> Optional[str]:
    """Find the cognee-mcp command path."""
    return shutil.which("cognee-mcp")


def _detect_llm_key() -> Optional[str]:
    """Check common env vars for an LLM API key."""
    for var in ("LLM_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        if os.environ.get(var):
            return var
    return None


def _mcp_server_config() -> dict:
    """Build the MCP server entry for cognee."""
    mcp_path = _find_cognee_mcp_path()

    if mcp_path:
        return {
            "command": "cognee-mcp",
            "args": ["--transport", "stdio"],
        }

    # Fallback: try uv run from the cognee-mcp directory
    return {
        "command": "uv",
        "args": ["--directory", "<path-to-cognee-mcp>", "run", "cognee-mcp"],
    }


def _get_mcp_config_path(target: str) -> Path:
    """Return the config file path for a given MCP target."""
    if target == "cursor":
        return Path.cwd() / ".cursor" / "mcp.json"
    elif target == "claude":
        if os.name == "nt":
            base = Path(os.environ.get("APPDATA", ""))
        else:
            base = Path.home() / "Library" / "Application Support"
        return base / "Claude" / "claude_desktop_config.json"
    elif target == "vscode":
        return Path.cwd() / ".vscode" / "mcp.json"
    else:
        raise ValueError(f"Unknown MCP target: {target}")


def _write_mcp_config(target: str, dry_run: bool = False) -> Path:
    """Write or update an MCP config file for the given target."""
    config_path = _get_mcp_config_path(target)
    server_entry = _mcp_server_config()

    # Read existing config if it exists
    existing = {}
    if config_path.exists():
        try:
            existing = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            existing = {}

    # Merge: add cognee-memory server without clobbering other servers
    if "mcpServers" not in existing:
        existing["mcpServers"] = {}

    existing["mcpServers"]["cognee"] = server_entry

    if dry_run:
        return config_path

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(existing, indent=2) + "\n")
    return config_path


def _print_tools_instructions(target: str) -> None:
    """Print integration instructions for a tools target."""
    instructions = {
        "openai": """
  Add this to your agent:

    from cognee.tools import for_openai, handle_tool_call

    tools = for_openai()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
    )

    if response.choices[0].message.tool_calls:
        for tool_call in response.choices[0].message.tool_calls:
            result = await handle_tool_call(tool_call)
            messages.append({
                "role": "tool",
                "content": result,
                "tool_call_id": tool_call.id,
            })""",
        "anthropic": """
  Add this to your agent:

    from cognee.tools import for_anthropic, handle_tool_call

    tools = for_anthropic()

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        messages=messages,
        tools=tools,
    )

    for block in response.content:
        if block.type == "tool_use":
            result = await handle_tool_call(block)""",
        "langchain": """
  Add this to your agent:

    from cognee.tools import for_langchain

    tools = for_langchain()

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)

    result = await executor.ainvoke({"input": "What do you remember?"})""",
        "crewai": """
  Add this to your agent:

    from cognee.tools import for_crewai

    agent = Agent(
        role="Assistant",
        goal="Help the user with memory-aware responses",
        tools=for_crewai(),
    )""",
    }

    fmt.echo(instructions.get(target, f"  No instructions available for '{target}'."))
    fmt.echo("")
    fmt.note(f"See full example: https://docs.cognee.ai/attach/{target}")


class AttachCommand(SupportsCliCommand):
    command_string = "attach"
    help_string = "Attach Cognee memory to your agent stack"
    docs_url = DEFAULT_DOCS_URL
    description = """
Attach Cognee memory to your existing agent stack.

Supports MCP clients (Cursor, Claude Desktop, VS Code) and
tool-based frameworks (OpenAI, LangChain, CrewAI, Anthropic).

For MCP targets, writes a config file so the client can start the Cognee MCP server.
For tool targets, prints the code snippet to add to your agent.
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--to",
            dest="target",
            choices=sorted(ALL_TARGETS),
            help="Target to attach to (e.g., cursor, openai, langchain)",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            default=False,
            help="Show what would be changed without writing anything",
        )

    def execute(self, args: argparse.Namespace) -> None:
        target = args.target

        if target is None:
            self._interactive_attach(args)
            return

        # Check LLM key
        key_var = _detect_llm_key()
        if key_var:
            fmt.echo(f"  LLM API Key? [found {key_var}] ok")
        else:
            fmt.warning(
                "No LLM API key found in environment. "
                "Set LLM_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY."
            )

        fmt.echo("")

        if target in MCP_TARGETS:
            self._attach_mcp(target, dry_run=args.dry_run)
        elif target in TOOLS_TARGETS:
            self._attach_tools(target)
        else:
            raise CliCommandException(
                f"Unknown target: {target}. Choose from: {', '.join(sorted(ALL_TARGETS))}",
                error_code=1,
            )

    def _attach_mcp(self, target: str, dry_run: bool = False) -> None:
        """Attach via MCP config file."""
        config_path = _write_mcp_config(target, dry_run=dry_run)

        if dry_run:
            fmt.echo(f"  Would write: {config_path}")
            fmt.echo("")
            fmt.note("No files were written. Run without --dry-run to apply.")
            return

        fmt.success(f"Wrote {config_path}")
        fmt.echo("")

        restart_msg = {
            "cursor": "Restart Cursor.",
            "claude": "Restart Claude Desktop.",
            "vscode": "Restart VS Code.",
        }
        fmt.echo(f"  {restart_msg.get(target, 'Restart your editor.')}")
        fmt.echo("  Your agent now has access to Cognee memory tools.")

    def _attach_tools(self, target: str) -> None:
        """Attach via tool import instructions."""
        _print_tools_instructions(target)

    def _interactive_attach(self, args: argparse.Namespace) -> None:
        """Interactive mode: detect environment and ask user."""
        fmt.echo("  Detecting environment...")
        fmt.echo("")

        # Detect MCP clients
        detected_mcp = []
        if (Path.cwd() / ".cursor").is_dir():
            detected_mcp.append("cursor")
        if (Path.cwd() / ".vscode").is_dir():
            detected_mcp.append("vscode")
        claude_config = _get_mcp_config_path("claude")
        if claude_config.parent.is_dir():
            detected_mcp.append("claude")

        # Detect Python frameworks
        detected_tools = []
        for req_file in ["requirements.txt", "pyproject.toml", "setup.py"]:
            req_path = Path.cwd() / req_file
            if req_path.exists():
                try:
                    content = req_path.read_text().lower()
                    if "openai" in content:
                        detected_tools.append("openai")
                    if "langchain" in content:
                        detected_tools.append("langchain")
                    if "crewai" in content:
                        detected_tools.append("crewai")
                    if "anthropic" in content:
                        detected_tools.append("anthropic")
                except OSError:
                    pass

        if detected_mcp:
            fmt.echo(f"  Detected MCP clients: {', '.join(detected_mcp)}")
        if detected_tools:
            fmt.echo(f"  Detected frameworks: {', '.join(detected_tools)}")
        if not detected_mcp and not detected_tools:
            fmt.echo("  No specific environment detected.")

        fmt.echo("")

        # Build choices
        choices = []
        for t in detected_mcp:
            choices.append(t)
        for t in detected_tools:
            if t not in choices:
                choices.append(t)

        if not choices:
            choices = sorted(ALL_TARGETS)

        # Ask user
        fmt.echo("  Available targets:")
        for i, choice in enumerate(choices, 1):
            label = "MCP" if choice in MCP_TARGETS else "Tools"
            fmt.echo(f"    {i}. {choice} ({label})")

        fmt.echo("")
        selection = fmt.prompt("  Choose a target (number or name)", default="1")

        # Resolve selection
        try:
            idx = int(selection) - 1
            if 0 <= idx < len(choices):
                target = choices[idx]
            else:
                raise ValueError()
        except ValueError:
            target = selection.strip().lower()

        if target not in ALL_TARGETS:
            raise CliCommandException(
                f"Unknown target: {target}",
                error_code=1,
            )

        # Delegate
        args.target = target
        self.execute(args)
