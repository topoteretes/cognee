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

export function installUv(os: PreferredOs): string {
  return os === "windows" ? "irm https://astral.sh/uv/install.ps1 | iex" : "curl -LsSf https://astral.sh/uv/install.sh | sh";
}

export function whichCommand(os: PreferredOs, bin: string): string {
  return os === "windows" ? `Get-Command ${bin}` : `which ${bin}`;
}
