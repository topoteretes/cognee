import argparse
import os
import stat
import subprocess
from pathlib import Path

from cognee.cli.reference import SupportsCliCommand
from cognee.cli import DEFAULT_DOCS_URL
import cognee.cli.echo as fmt
from cognee.cli.exceptions import CliCommandException, CliCommandInnerException

_MARK_START = "# >>> cognee post-commit hook (managed) >>>"
_MARK_END = "# <<< cognee post-commit hook (managed) <<<"


def _hooks_dir(start: Path) -> Path:
    """Resolve the git hooks directory for the repo containing ``start``.

    Asks git directly (``git rev-parse --git-path hooks``) so this honors
    ``core.hooksPath`` and works in worktrees/submodules where ``.git`` is a file,
    not a directory. Falls back to a plain ``.git/hooks`` walk if the git CLI
    isn't available.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-path", "hooks"],
            cwd=str(start),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            hooks = Path(result.stdout.strip())
            return hooks if hooks.is_absolute() else (start / hooks).resolve()
    except (OSError, subprocess.SubprocessError):
        pass

    for candidate in (start, *start.parents):
        if (candidate / ".git").is_dir():
            return candidate / ".git" / "hooks"
    raise CliCommandInnerException(
        "Not inside a git repository. Run `cognee hook install` from within your project."
    )


def _managed_block(path: str, dataset_name: str) -> str:
    # `|| true` keeps a commit from failing if cognee/the server is unavailable.
    return (
        f"{_MARK_START}\n"
        f'cognee update "{path}" --dataset-name "{dataset_name}" || true\n'
        f"{_MARK_END}\n"
    )


class HookCommand(SupportsCliCommand):
    command_string = "hook"
    help_string = "Install or remove a git post-commit hook that refreshes the graph"
    docs_url = DEFAULT_DOCS_URL
    description = """
Manage a git post-commit hook that keeps your Cognee graph in sync with your repo.

    cognee hook install                 # refresh from repo root into main_dataset on commit
    cognee hook install --path ./docs --dataset-name my_project
    cognee hook uninstall

`install` adds a small managed block to `.git/hooks/post-commit` that runs
`cognee update`. An existing hook is preserved — only the managed block is added or
replaced. `uninstall` removes just that block.
    """

    def configure_parser(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "action",
            choices=["install", "uninstall"],
            help="Install or uninstall the post-commit hook",
        )
        parser.add_argument(
            "--path",
            default=".",
            help="Path passed to `cognee update` on each commit (default: repo root)",
        )
        parser.add_argument(
            "--dataset-name",
            "-d",
            default="main_dataset",
            help="Dataset to update on each commit (default: main_dataset)",
        )

    def execute(self, args: argparse.Namespace) -> None:
        try:
            hooks_dir = _hooks_dir(Path.cwd())
            hook_path = hooks_dir / "post-commit"
            existing = hook_path.read_text(encoding="utf-8") if hook_path.exists() else ""
            # Strip any previously-managed block so install/uninstall are idempotent.
            stripped = self._strip_block(existing)

            if args.action == "install":
                block = _managed_block(args.path, args.dataset_name)
                if stripped.strip():
                    content = stripped.rstrip("\n") + "\n\n" + block
                else:
                    content = "#!/bin/sh\n" + block
                hooks_dir.mkdir(parents=True, exist_ok=True)
                hook_path.write_text(content, encoding="utf-8")
                self._make_executable(hook_path)
                fmt.success(
                    f"Installed post-commit hook at {hook_path} "
                    f"(updates '{args.dataset_name}' from '{args.path}')."
                )
            else:
                if not hook_path.exists() or stripped == existing:
                    fmt.echo("No cognee-managed post-commit hook found; nothing to remove.")
                    return
                if stripped.strip() in ("", "#!/bin/sh"):
                    hook_path.unlink()  # our block was the only content
                else:
                    hook_path.write_text(stripped.rstrip("\n") + "\n", encoding="utf-8")
                fmt.success("Removed the cognee-managed post-commit hook.")

        except CliCommandInnerException as e:
            raise CliCommandException(str(e), error_code=1) from e
        except Exception as e:
            raise CliCommandException(f"Failed to manage hook: {str(e)}", error_code=1) from e

    @staticmethod
    def _strip_block(content: str) -> str:
        if _MARK_START not in content or _MARK_END not in content:
            return content
        start = content.index(_MARK_START)
        end = content.index(_MARK_END) + len(_MARK_END)
        # Also swallow a trailing newline left by the removed block.
        after = content[end:]
        if after.startswith("\n"):
            after = after[1:]
        return content[:start] + after

    @staticmethod
    def _make_executable(path: Path) -> None:
        try:
            mode = path.stat().st_mode
            path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        except OSError:
            # Filesystems without executable bits (e.g. Windows) — git still runs it
            # via its bundled shell, so this is best-effort.
            pass
