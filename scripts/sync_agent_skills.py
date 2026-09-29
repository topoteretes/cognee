"""Mirror .claude/skills/ into .agents/skills/ so Codex sees the same skills.

Claude Code reads skills from .claude/skills/ and Codex only from
.agents/skills/. .claude/skills/ is the source; .agents/skills/ is a real
copy (not a symlink, which Windows checkouts turn into a text file) and is
never edited by hand.

Runs as a pre-commit hook, locally and in CI (pre_test.yml). It rewrites
.agents/skills/ to match the source and exits 1 when it had to change
anything, so a hand edit or a missed sync fails the check.
"""

import filecmp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / ".claude" / "skills"
MIRROR = ROOT / ".agents" / "skills"


def _files(directory: Path) -> set[Path]:
    if not directory.is_dir():
        return set()
    return {path.relative_to(directory) for path in directory.rglob("*") if path.is_file()}


def sync() -> list[str]:
    """Make MIRROR an exact copy of SOURCE and return the paths it changed."""
    changed = []

    # A symlink or a plain file where the directory should be (a Windows
    # checkout of the old symlink) is replaced by a real directory.
    if MIRROR.is_symlink() or (MIRROR.exists() and not MIRROR.is_dir()):
        MIRROR.unlink()
        changed.append(str(MIRROR.relative_to(ROOT)))

    source_files = _files(SOURCE)
    mirror_files = _files(MIRROR)

    for relative in sorted(source_files):
        target = MIRROR / relative
        if relative in mirror_files and filecmp.cmp(SOURCE / relative, target, shallow=False):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / relative, target)
        changed.append(str(target.relative_to(ROOT)))

    for relative in sorted(mirror_files - source_files):
        (MIRROR / relative).unlink()
        changed.append(str((MIRROR / relative).relative_to(ROOT)))

    # Drop directories a removed skill left empty.
    for directory in sorted(MIRROR.rglob("*"), reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()

    return changed


def main() -> int:
    if not SOURCE.is_dir():
        print(f"{SOURCE.relative_to(ROOT)} not found", file=sys.stderr)
        return 1

    changed = sync()
    if not changed:
        return 0

    print(".agents/skills/ is generated from .claude/skills/ and was out of date. Updated:")
    for path in changed:
        print(f"  {path}")
    print("Edit skills in .claude/skills/ only, then stage .agents/skills/ and commit again.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
