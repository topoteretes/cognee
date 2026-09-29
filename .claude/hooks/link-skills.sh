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
cmd //c mklink /J "$(cygpath -w "$PWD/$link")" "$(cygpath -w "$PWD/$target")" >/dev/null 2>&1 || exit 0

# The tracked symlink entry now differs from the worktree; keep git status clean.
git update-index --skip-worktree "$link" 2>/dev/null
exclude="$(git rev-parse --git-path info/exclude 2>/dev/null)"
if [ -n "$exclude" ] && ! grep -qxF "/$link/" "$exclude" 2>/dev/null; then
  mkdir -p "$(dirname "$exclude")" && echo "/$link/" >>"$exclude"
fi

echo '{"hookSpecificOutput":{"hookEventName":"SessionStart","reloadSkills":true}}'
