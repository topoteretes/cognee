"""Find the coding agents installed on this machine and the sessions each one has kept.

For Claude Code and Codex it reports how each is installed (command line, desktop app, VS Code
or Cursor extension) and how many sessions are on disk, split by where they were started:
every session file records that (`entrypoint` for Claude Code, `originator` for Codex). It also
counts the sessions per project folder (the session's working directory), so you can choose
which projects to import.
Apps whose history this cookbook cannot read are named too, with the reason, so nothing is
silently left out. Reads only the first lines of each session file; no LLM, no cognee.

Run alone: uv run python examples/cookbooks/import_agent_sessions/scripts/find_agents.py
"""

import json
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

try:
    from scripts.read_sessions import AGENTS, history_files
except ModuleNotFoundError:  # run alone from scripts/
    from read_sessions import AGENTS, history_files

HOME = Path.home()
APPLICATIONS = (Path("/Applications"), HOME / "Applications")

# Where each agent can be installed: a command on PATH, a macOS app, or an editor extension.
INSTALLS = {
    "claude": {
        "command line": "claude",
        "desktop app": "Claude.app",
        "VS Code extension": ".vscode/extensions/anthropic.claude-code-*",
        "Cursor extension": ".cursor/extensions/anthropic.claude-code-*",
    },
    "codex": {
        "command line": "codex",
        "desktop app": "Codex.app",
        "VS Code extension": ".vscode/extensions/openai.chatgpt-*",
        "Cursor extension": ".cursor/extensions/openai.chatgpt-*",
    },
}

# The session's own record of where it was started, in words.
SURFACES = {
    "cli": "terminal",
    "sdk-cli": "Agent SDK",
    "sdk-py": "Agent SDK",
    "sdk-ts": "Agent SDK",
    "claude-vscode": "IDE extension",
    "claude-desktop": "desktop app",
    "codex-tui": "terminal",
    "codex_cli_rs": "terminal",
    "codex_exec": "codex exec",
    "codex_vscode": "IDE extension",
    "Codex Desktop": "desktop app",
}

# Installed apps that keep chats this cookbook does not read, and why.
NOT_READ = {
    "Claude.app": "Claude desktop chats are stored in your Claude account, not on this machine"
    " (claude.ai > Settings > Privacy > Export data gives you conversations.json)",
    "Cursor.app": "Cursor's own agent keeps its chats in Cursor's database, which this cookbook"
    " does not read (Claude Code or Codex running inside Cursor are imported)",
    "ChatGPT.app": "ChatGPT chats are stored in your OpenAI account, not on this machine",
}


@dataclass
class Agent:
    name: str  # "claude" or "codex"
    installs: list[str] = field(default_factory=list)
    surfaces: Counter = field(default_factory=Counter)
    folders: Counter = field(default_factory=Counter)  # project folder -> sessions

    @property
    def sessions(self) -> int:
        return sum(self.surfaces.values())


def installed_app(name: str) -> bool:
    return any((folder / name).exists() for folder in APPLICATIONS)


def installs(agent: str) -> list[str]:
    found = []
    for label, target in INSTALLS[agent].items():
        if label == "command line":
            hit = shutil.which(target)
        elif label == "desktop app":
            hit = installed_app(target)
        else:
            hit = any(HOME.glob(target))
        if hit:
            found.append(label)
    return found


def origin(path: Path, agent: str) -> tuple[str, str]:
    """Where one session was started and its project folder, from the first lines of its file."""
    where, folder = "", ""
    with path.open(encoding="utf-8", errors="replace") as file:
        for _, line in zip(range(40), file):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            source = record if agent == "claude" else record.get("payload") or {}
            value = source.get("entrypoint" if agent == "claude" else "originator")
            where = where or SURFACES.get(value, value) or ""
            folder = folder or source.get("cwd") or ""
            if where and folder:
                break
    return where or "unknown", folder or "unknown"


def find_agents() -> list[Agent]:
    found = []
    for name in AGENTS:
        agent = Agent(name, installs(name))
        for path in history_files(name):
            where, folder = origin(path, name)
            agent.surfaces[where] += 1
            agent.folders[folder] += 1
        found.append(agent)
    return found


def projects(agents: list[Agent]) -> list[str]:
    """One line per project folder, most sessions first: the folder and each agent's count."""
    folders = Counter()
    for agent in agents:
        folders.update(agent.folders)
    return [
        folder
        + " ("
        + ", ".join(
            f"{AGENTS[agent.name]} {agent.folders[folder]}"
            for agent in agents
            if agent.folders[folder]
        )
        + ")"
        for folder, _ in folders.most_common()
    ]


def not_read() -> list[str]:
    """One line per installed app whose history this cookbook leaves out."""
    return [reason for app, reason in NOT_READ.items() if installed_app(app)]


def describe(agent: Agent) -> str:
    how = ", ".join(agent.installs) if agent.installs else "not installed"
    if not agent.sessions:
        return f"{AGENTS[agent.name]} ({how}): no sessions on this machine"
    split = ", ".join(f"{count} {where}" for where, count in agent.surfaces.most_common())
    return f"{AGENTS[agent.name]} ({how}): {agent.sessions} sessions ({split})"


if __name__ == "__main__":
    agents = find_agents()
    for agent in agents:
        print(f"[find_agents] {describe(agent)}")
    for line in projects(agents):
        print(f"[find_agents] Project: {line}")
    for line in not_read():
        print(f"[find_agents] Not imported: {line}.")
