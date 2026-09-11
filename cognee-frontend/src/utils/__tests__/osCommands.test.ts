import { curlBin, exportEnvVar, homePath, installUv, whichCommand, writeSkillFile } from "../osCommands";

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
