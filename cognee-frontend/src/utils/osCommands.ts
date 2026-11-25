import { PreferredOs } from "@/ui/layout/OsPreferenceContext";

export function exportEnvVar(os: PreferredOs, name: string, value: string): string {
  return os === "windows" ? `$env:${name} = "${value}"` : `export ${name}="${value}"`;
}

// posixRelPath must start with "/", e.g. "/.claude/skills/cognee/SKILL.md"
export function homePath(os: PreferredOs, posixRelPath: string): string {
  if (os === "windows") return `$env:USERPROFILE${posixRelPath.replace(/\//g, "\\")}`;
  return `~${posixRelPath}`;
}

export function curlBin(os: PreferredOs): string {
  return os === "windows" ? "curl.exe" : "curl";
}

// dirPosixRelPath must start with "/", e.g. "/.claude/skills/cognee"
export function writeSkillFile(os: PreferredOs, dirPosixRelPath: string, fileName: string, content: string): string {
  if (os === "windows") {
    const dirPath = `$env:USERPROFILE${dirPosixRelPath.replace(/\//g, "\\")}`;
    return `New-Item -ItemType Directory -Force -Path "${dirPath}" | Out-Null\nSet-Content -Path "${dirPath}\\${fileName}" -Value @'\n${content}\n'@`;
  }
  const dirPath = `~${dirPosixRelPath}`;
  return `mkdir -p ${dirPath} && cat > ${dirPath}/${fileName} << 'COGNEE_EOF'\n${content}\nCOGNEE_EOF`;
}

// Path of the env file both the Claude Code and Codex plugins read (CLO-532).
export const COGNEE_ENV_FILE_POSIX = "/.cognee/.env";

// One-time credential setup for the agent plugins: writes ~/.cognee/.env, which
// both plugins read at session start. Replaces the old per-terminal `export`,
// which was lost the moment the shell closed.
//
// Re-runnable by design — onboarding can be replayed. Existing lines for the
// same keys are filtered out and the new values appended, so the file never
// accumulates duplicates. Unrelated keys (notably LLM_API_KEY, which selects
// local mode) are preserved, and an `export ` prefix on an old line is matched
// too, since the plugin accepts that form.
//
// Deliberately a SINGLE line per OS: it is shown to the user in a combined
// setup block, where a multi-line heredoc read as a wall of shell.
//
// Both branches embed the values inside shell single quotes, so a value
// containing one would otherwise close the quote and hand the rest of itself to
// the shell as code. Escaped here rather than trusted, since the values arrive
// from a backend response and an env var.

/** Ends the single-quoted run, emits a literal quote, reopens it. */
function shQuote(value: string): string {
  return value.replace(/'/g, `'\\''`);
}

/**
 * The value is emitted inside a double-quoted dotenv line, so a literal " would
 * close it and leave a malformed entry. Not a shell concern — escaped for the
 * reader, before either shell quoting runs.
 */
function dotenvValue(value: string): string {
  // Backslash first: escaping the quotes first would leave the backslash this
  // adds open to being escaped again on the second pass.
  return value.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

/** PowerShell's only escape inside a single-quoted string is doubling it. */
function psQuote(value: string): string {
  return value.replace(/'/g, `''`);
}

/**
 * The key names go into the alternation of the filter pattern that decides
 * which existing lines to drop. Both callers pass COGNEE_BASE_URL and
 * COGNEE_API_KEY, which are already inert here, so this changes nothing today —
 * it is a guard on an exported utility, so a future caller cannot pass a name
 * whose metacharacters widen the alternation and strip unrelated keys out of a
 * file this function then overwrites.
 */
function reQuote(name: string): string {
  return name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function writeCogneeEnvFile(os: PreferredOs, vars: Record<string, string>): string {
  const names = Object.keys(vars);
  const alternation = names.map(reQuote).join("|");

  if (os === "windows") {
    const dir = `$env:USERPROFILE\\.cognee`;
    // -cnotmatch, not -notmatch: PowerShell's default is case-insensitive while
    // the posix branch's grep -vE is not, so the two would disagree on a
    // lowercase `cognee_api_key=` line.
    const pattern = `'^\\s*(export\\s+)?(${alternation})='`;
    const newLines = names.map((n) => `'${n}="${psQuote(dotenvValue(vars[n]))}"'`).join(", ");
    // Windows has no chmod: the file inherits the profile directory's ACL, which
    // on a shared machine can be readable by other accounts. icacls drops the
    // inherited entries and grants the current user alone — the counterpart of
    // the chmod 600 on the branch below, for a file holding an API key.
    //
    // Same write-temp / restrict / move shape as the posix branch, for the same
    // two reasons: the key never exists under the inherited ACL (icacls runs on
    // the temp file, before it becomes .env), and the replace is atomic, so an
    // interrupted run cannot leave a half-written file — which would take the
    // user's unrelated keys, LLM_API_KEY among them, with it. Move-Item within
    // one volume keeps the explicit ACL, so the restriction survives the move.
    // The D in (R,W,D) is what lets Move-Item rename the temp file and replace
    // an existing .env a previous run already restricted, instead of leaning on
    // FILE_DELETE_CHILD being inherited from %USERPROFILE%.
    // if ($?) is the `&&` of the posix branch: without it PowerShell's `;` would
    // move a partial temp file over a good .env when Set-Content fails.
    const restrict = `icacls $t /inheritance:r /grant:r "$($env:USERNAME):(R,W,D)" | Out-Null`;
    return `New-Item -ItemType Directory -Force "${dir}" | Out-Null; $f = "${dir}\\.env"; $t = "$f.new"; @(if (Test-Path $f) { Get-Content $f | Where-Object { $_ -cnotmatch ${pattern} } }) + @(${newLines}) | Set-Content $t; if ($?) { ${restrict}; Move-Item -Force $t $f }`;
  }

  const file = "~/.cognee/.env";
  const values = names.map((n) => `'${n}="${shQuote(dotenvValue(vars[n]))}"'`).join(" ");
  // 2>/dev/null covers the first run, where the file does not exist yet: grep's
  // failure is discarded and the group still exits on printf's success.
  const keep = `grep -vE '^[[:space:]]*(export[[:space:]]+)?(${alternation})=' ${file} 2>/dev/null`;
  // umask 077 in a subshell: the redirect creates .env.new under the caller's
  // umask, so without it the API key sits world-readable until the chmod after
  // the mv. The Windows branch above has no equivalent step — %USERPROFILE% is
  // user-restricted by default, and icacls tightens it there instead.
  return `mkdir -p ~/.cognee && ( umask 077; { ${keep}; printf '%s\\n' ${values}; } > ${file}.new ) && mv ${file}.new ${file} && chmod 600 ${file}`;
}

export function installUv(os: PreferredOs): string {
  return os === "windows" ? "irm https://astral.sh/uv/install.ps1 | iex" : "curl -LsSf https://astral.sh/uv/install.sh | sh";
}

export function whichCommand(os: PreferredOs, bin: string): string {
  return os === "windows" ? `Get-Command ${bin}` : `which ${bin}`;
}
