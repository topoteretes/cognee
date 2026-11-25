import { curlBin, exportEnvVar, homePath, installUv, whichCommand, writeSkillFile, writeCogneeEnvFile, COGNEE_ENV_FILE_POSIX } from "@/utils/osCommands";

const VARS = { COGNEE_BASE_URL: "https://t1.aws.cognee.ai", COGNEE_API_KEY: "ck_abc" };

describe("osCommands", () => {
  it("exportEnvVar uses export on mac and $env: assignment on windows", () => {
    expect(exportEnvVar("mac", "COGNEE_API_KEY", "abc")).toBe(`export COGNEE_API_KEY="abc"`);
    expect(exportEnvVar("windows", "COGNEE_API_KEY", "abc")).toBe(`$env:COGNEE_API_KEY = "abc"`);
  });

  it("homePath expands to a PowerShell-native path on windows, not %USERPROFILE%", () => {
    expect(homePath("mac", "/.claude/skills/cognee/SKILL.md")).toBe("~/.claude/skills/cognee/SKILL.md");
    expect(homePath("windows", "/.claude/skills/cognee/SKILL.md")).toBe(
      "$env:USERPROFILE\\.claude\\skills\\cognee\\SKILL.md",
    );
  });

  it("curlBin uses curl.exe only on windows", () => {
    expect(curlBin("mac")).toBe("curl");
    expect(curlBin("windows")).toBe("curl.exe");
  });

  it("installUv uses the PowerShell installer only on windows", () => {
    expect(installUv("mac")).toBe("curl -LsSf https://astral.sh/uv/install.sh | sh");
    expect(installUv("windows")).toBe("irm https://astral.sh/uv/install.ps1 | iex");
  });

  it("whichCommand uses Get-Command only on windows", () => {
    expect(whichCommand("mac", "uvx")).toBe("which uvx");
    expect(whichCommand("windows", "uvx")).toBe("Get-Command uvx");
  });

  it("writeSkillFile emits a PowerShell here-string on windows and a bash heredoc on mac", () => {
    const mac = writeSkillFile("mac", "/.claude/skills/cognee", "SKILL.md", "content");
    expect(mac).toBe(
      "mkdir -p ~/.claude/skills/cognee && cat > ~/.claude/skills/cognee/SKILL.md << 'COGNEE_EOF'\ncontent\nCOGNEE_EOF",
    );

    const windows = writeSkillFile("windows", "/.claude/skills/cognee", "SKILL.md", "content");
    expect(windows).toBe(
      'New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\\.claude\\skills\\cognee" | Out-Null\n' +
        'Set-Content -Path "$env:USERPROFILE\\.claude\\skills\\cognee\\SKILL.md" -Value @\'\ncontent\n\'@',
    );
  });
});

describe("writeCogneeEnvFile", () => {
  describe("unix", () => {
    const cmd = writeCogneeEnvFile("mac", VARS);

    it("targets the env file both plugins read", () => {
      expect(COGNEE_ENV_FILE_POSIX).toBe("/.cognee/.env");
      expect(cmd).toContain("~/.cognee/.env");
    });

    it("writes both credential values", () => {
      expect(cmd).toContain('COGNEE_BASE_URL="https://t1.aws.cognee.ai"');
      expect(cmd).toContain('COGNEE_API_KEY="ck_abc"');
    });

    it("strips prior values for the same keys so re-running never duplicates", () => {
      expect(cmd).toContain("grep -vE '^[[:space:]]*(export[[:space:]]+)?(COGNEE_BASE_URL|COGNEE_API_KEY)='");
    });

    it("tolerates the file not existing yet, which is the first-run case", () => {
      // Without this, grep's "no such file" failure surfaces to the user on the
      // very first run.
      expect(cmd).toContain("2>/dev/null");
    });

    it("stays on one line so it reads as a single command in the setup block", () => {
      expect(cmd).not.toContain("\n");
    });

    it("restricts the file to the owner", () => {
      expect(cmd).toContain("chmod 600 ~/.cognee/.env");
    });

    it("creates the temp file already private, not just the final one", () => {
      // Without the umask the redirect makes .env.new under the caller's umask,
      // leaving the API key world-readable on a umask 022 box until the chmod.
      expect(cmd).toContain("( umask 077;");
    });

    it("does not append to the live file", () => {
      // Appending is what CLO-532's original block did; replay would grow the
      // file on every run.
      expect(cmd).not.toContain(">> ~/.cognee/.env\n");
      expect(cmd).toContain("mv ~/.cognee/.env.new ~/.cognee/.env");
    });
  });

  describe("windows", () => {
    const cmd = writeCogneeEnvFile("windows", VARS);

    it("uses the PowerShell profile path, not a posix one", () => {
      expect(cmd).toContain("$env:USERPROFILE\\.cognee");
      expect(cmd).not.toContain("~/.cognee");
    });

    it("writes both credential values", () => {
      expect(cmd).toContain(`'COGNEE_BASE_URL="https://t1.aws.cognee.ai"'`);
      expect(cmd).toContain(`'COGNEE_API_KEY="ck_abc"'`);
    });

    it("filters prior values for the same keys", () => {
      expect(cmd).toContain("-cnotmatch '^\\s*(export\\s+)?(COGNEE_BASE_URL|COGNEE_API_KEY)='");
    });

    it("handles a not-yet-existing file", () => {
      expect(cmd).toContain("if (Test-Path $f)");
    });

    it("uses no bash-only syntax", () => {
      expect(cmd).not.toContain("mkdir -p");
      expect(cmd).not.toContain("chmod");
      expect(cmd).not.toContain("<<'EOF'");
    });

    // The counterpart of the unix branch's chmod 600: the file holds an API key
    // and would otherwise inherit a profile ACL other accounts can read.
    it("restricts the file to the current user", () => {
      expect(cmd).toContain(`icacls $t /inheritance:r /grant:r "$($env:USERNAME):(R,W,D)"`);
    });

    it("restricts the temp file before it becomes .env, as the posix branch does", () => {
      // Restricting after the move would leave the key under the inherited ACL
      // for that window — the Windows equivalent of writing .env.new under the
      // caller's umask.
      expect(cmd.indexOf("icacls $t")).toBeLessThan(cmd.indexOf("Move-Item"));
    });

    it("replaces .env atomically instead of overwriting it in place", () => {
      // Set-Content straight onto $f is not atomic: an interrupted run leaves a
      // half-written file, taking unrelated keys like LLM_API_KEY with it.
      expect(cmd).toContain("Set-Content $t");
      expect(cmd).not.toContain("Set-Content $f");
      expect(cmd).toContain("Move-Item -Force $t $f");
    });

    it("does not move the temp file over .env when the write failed", () => {
      // PowerShell's `;` runs the next statement regardless, so without the
      // guard a failed Set-Content would still clobber a good .env. This is the
      // `&&` of the posix branch.
      expect(cmd).toContain("Set-Content $t; if ($?) {");
    });

    it("stays on one line so it reads as a single command in the setup block", () => {
      expect(cmd).not.toContain("\n");
    });
  });

  // A quote in a value would otherwise close the single-quoted run the value
  // sits in and hand the remainder to the shell as code.
  describe("values containing shell metacharacters", () => {
    const HOSTILE = { COGNEE_API_KEY: `a'$(id)'b` };

    it("escapes single quotes for bash without dropping any character", () => {
      const cmd = writeCogneeEnvFile("mac", HOSTILE);
      expect(cmd).toContain(`'COGNEE_API_KEY="a'\\''$(id)'\\''b"'`);
      expect(cmd).not.toContain(`"a'$(id)'b"`);
    });

    it("escapes single quotes for PowerShell by doubling them", () => {
      const cmd = writeCogneeEnvFile("windows", HOSTILE);
      expect(cmd).toContain(`'COGNEE_API_KEY="a''$(id)''b"'`);
    });

    it("escapes a double quote so the dotenv line stays well formed", () => {
      // The value sits inside a double-quoted entry; an unescaped " would end it
      // early and leave the rest of the key as garbage for the plugin's reader.
      expect(writeCogneeEnvFile("mac", { COGNEE_API_KEY: 'ck_a"b' }))
        .toContain(`'COGNEE_API_KEY="ck_a\\"b"'`);
      expect(writeCogneeEnvFile("windows", { COGNEE_API_KEY: 'ck_a"b' }))
        .toContain(`'COGNEE_API_KEY="ck_a\\"b"'`);
    });

    it("escapes a backslash so the dotenv reader does not treat it as an escape", () => {
      // Without this the backslash would consume the character after it, and a
      // trailing one would escape the closing quote of the entry.
      expect(writeCogneeEnvFile("mac", { COGNEE_API_KEY: "ck_a\\b" }))
        .toContain(`'COGNEE_API_KEY="ck_a\\\\b"'`);
    });

    it("leaves characters that are literal inside single quotes untouched", () => {
      // Backticks, $ and ; carry no meaning in a single-quoted run, so escaping
      // them would corrupt the value that lands in the file.
      const cmd = writeCogneeEnvFile("mac", { COGNEE_API_KEY: "a`b$c;d" });
      expect(cmd).toContain(`'COGNEE_API_KEY="a\`b$c;d"'`);
    });
  });

  // The names land in the alternation of the pattern that decides which existing
  // lines to drop, and the file is then overwritten from what survives — an
  // unescaped metacharacter would widen the match and take unrelated keys with
  // it. No caller does this today; the guard is on the exported utility.
  describe("key names containing regex metacharacters", () => {
    const ODD = { "TEST|EVIL_KEY": "v" };

    it("escapes them in the bash pattern so the alternation cannot widen", () => {
      const cmd = writeCogneeEnvFile("mac", ODD);
      expect(cmd).toContain("(TEST\\|EVIL_KEY)=");
    });

    it("escapes them in the PowerShell pattern too", () => {
      const cmd = writeCogneeEnvFile("windows", ODD);
      expect(cmd).toContain("(TEST\\|EVIL_KEY)=");
    });

    it("leaves the ordinary key names byte-identical", () => {
      // Both real callers pass these two, so the escaping must be a no-op for
      // them — the emitted command has to stay exactly what shipped.
      expect(writeCogneeEnvFile("mac", VARS)).toContain("(COGNEE_BASE_URL|COGNEE_API_KEY)=");
      expect(writeCogneeEnvFile("windows", VARS)).toContain("(COGNEE_BASE_URL|COGNEE_API_KEY)=");
    });
  });

  it("keeps unrelated keys by filtering only the keys it writes", () => {
    // LLM_API_KEY selects local mode and lives in the same shared file — a
    // blanket truncate would silently break local-mode users.
    const cmd = writeCogneeEnvFile("mac", VARS);
    expect(cmd).not.toContain("LLM_API_KEY");
    expect(cmd).not.toContain("> ~/.cognee/.env\n");
  });
});
