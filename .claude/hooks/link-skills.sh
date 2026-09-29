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

cd "${CLAUDE_PROJECT_DIR:-.}" 2>/dev/null || exit 0

link=".claude/skills"
target=".agents/skills"

[ -d "$link" ] && exit 0 # symlink or junction already resolves
[ -d "$target" ] || exit 0

case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN*) ;;
  *) exit 0 ;;
esac

rm -f "$link" 2>/dev/null # the text placeholder, or a junction left dangling by a moved checkout

# PowerShell rather than `cmd //c mklink /J`: Git Bash rewrites the `/J` flag
# into a drive path. Paths travel as env vars so spaces and quotes are safe.
if ! SKILLS_LINK="$(cygpath -w "$PWD/$link")" SKILLS_TARGET="$(cygpath -w "$PWD/$target")" \
  powershell.exe -NoProfile -NonInteractive -Command \
  'New-Item -ItemType Junction -Path $env:SKILLS_LINK -Target $env:SKILLS_TARGET | Out-Null' >&2; then
  echo "link-skills: could not create the $link junction" >&2
  git checkout -- "$link" 2>/dev/null # put the placeholder back
  exit 0
fi

# The tracked symlink entry now differs from the worktree; keep git status clean.
git update-index --skip-worktree "$link" 2>/dev/null
exclude="$(git rev-parse --git-path info/exclude 2>/dev/null)"
if [ -n "$exclude" ] && ! grep -qxF "/$link/" "$exclude" 2>/dev/null; then
  mkdir -p "$(dirname "$exclude")" && echo "/$link/" >>"$exclude"
fi

echo '{"hookSpecificOutput":{"hookEventName":"SessionStart","reloadSkills":true}}'
