#!/usr/bin/env bash
# SessionStart hook: make .claude/skills point at .agents/skills on every OS.
#
# .agents/skills holds the real skill files (Codex reads it natively).
# .claude/skills is committed as a git symlink to it, which works on macOS,
# Linux and Windows clones with symlinks enabled. On a default Windows clone
# git checks the symlink out as a small text file instead, so this hook
# replaces that file with a directory junction (no admin or Developer Mode
# needed) and asks Claude Code to reload skills in the same session.
#
# The hook never fails the session: every error path exits 0.
#
# Windows only: the junction makes the worktree differ from the tracked symlink
# blob, so the hook marks the path skip-worktree. If .claude/skills ever changes
# upstream, git then refuses to update it ("Entry ... not uptodate"). Recover with
#   git update-index --no-skip-worktree .claude/skills && git checkout -- .claude/skills
# and start a new session; this hook rebuilds the junction.

cd "${CLAUDE_PROJECT_DIR:-.}" 2>/dev/null || exit 0

link=".claude/skills"
target=".agents/skills"

[ -d "$link" ] && exit 0 # symlink or junction already resolves
[ -d "$target" ] || exit 0

case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN*) ;;
  *) exit 0 ;;
esac

# Two things can sit here: the text placeholder git wrote instead of the symlink,
# or a junction left dangling by a moved checkout. MSYS surfaces a junction as a
# symlink, so `rm -f` unlinks either one without following it into
# .agents/skills; `rmdir` covers only a plain empty directory, since Git Bash
# refuses a junction with "Not a directory". Deliberately not `rm -rf`: it would
# not follow a junction either, but it would silently erase a real directory
# committed here -- which the check below is meant to report, not destroy.
rm -f "$link" 2>/dev/null || :
if [ -e "$link" ] || [ -L "$link" ]; then
  rmdir "$link" 2>/dev/null || :
fi
if [ -e "$link" ] || [ -L "$link" ]; then
  # Most likely a real directory someone committed here. Destroying it to make
  # room would be worse than starting the session without repo skills.
  echo "link-skills: $link exists and could not be removed; leaving it as is" >&2
  exit 0
fi

# PowerShell rather than `cmd //c mklink /J`: Git Bash rewrites the `/J` flag
# into a drive path. Paths travel as env vars so spaces and quotes are safe.
if ! SKILLS_LINK="$(cygpath -w "$PWD/$link")" SKILLS_TARGET="$(cygpath -w "$PWD/$target")" \
  powershell.exe -NoProfile -NonInteractive -Command \
  'New-Item -ItemType Junction -Path $env:SKILLS_LINK -Target $env:SKILLS_TARGET | Out-Null' >&2; then
  echo "link-skills: could not create the $link junction" >&2
  # Clear skip-worktree first: with the bit set from an earlier run, checkout is
  # a silent no-op and the placeholder never comes back.
  git update-index --no-skip-worktree "$link" 2>/dev/null
  git checkout -- "$link" 2>/dev/null # put the placeholder back
  exit 0
fi

# The tracked symlink entry now differs from the worktree; keep git status clean.
git update-index --skip-worktree "$link" 2>/dev/null
# Not redundant with skip-worktree: that hides the symlink entry itself, while
# the files now visible *under* the junction are untracked paths. Without this,
# `git status -uall` lists every skill and `git add -A` commits a second copy.
exclude="$(git rev-parse --git-path info/exclude 2>/dev/null)"
if [ -n "$exclude" ] && ! grep -qxF "/$link/" "$exclude" 2>/dev/null; then
  mkdir -p "$(dirname "$exclude")" && echo "/$link/" >>"$exclude"
fi

echo '{"hookSpecificOutput":{"hookEventName":"SessionStart","reloadSkills":true}}'
