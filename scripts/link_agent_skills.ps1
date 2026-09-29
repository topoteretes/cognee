<#
.SYNOPSIS
    Makes .agents\skills point at .claude\skills on Windows, so Codex finds the skills.

.DESCRIPTION
    Codex reads skills only from .agents\skills. In the repository that path is a
    symlink to ../.claude/skills. With git's default core.symlinks=false on Windows,
    the checkout holds a small text file there instead, and Codex sees no skills.

    This script replaces that text file with a directory junction to .claude\skills.
    Junctions need neither admin rights nor Developer Mode. It then tells git to
    ignore the local change (skip-worktree), so .agents\skills does not show up in
    git status or get committed.

    Run it once per clone, and again after moving the clone: a junction stores an
    absolute path. If symlinks already work (Developer Mode plus
    git config core.symlinks true), the script reports that and changes nothing.

.PARAMETER Undo
    Remove the junction and restore the file git expects.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\link_agent_skills.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\link_agent_skills.ps1 -Undo
#>
[CmdletBinding()]
param(
    [switch]$Undo
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$linkPath = Join-Path $repoRoot ".agents\skills"
$targetPath = Join-Path $repoRoot ".claude\skills"
$gitPath = ".agents/skills"

function Invoke-Git {
    & git -C $repoRoot @args
    if ($LASTEXITCODE -ne 0) {
        throw "git $($args -join ' ') failed with exit code $LASTEXITCODE"
    }
}

function Get-LinkItem {
    # -Force also returns hidden items; a missing path returns $null.
    Get-Item -LiteralPath $linkPath -Force -ErrorAction SilentlyContinue
}

function Test-IsReparsePoint($item) {
    $null -ne $item -and ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)
}

if (-not (Test-Path -LiteralPath $targetPath -PathType Container)) {
    throw "$targetPath not found. Run this script from a cognee checkout."
}

$item = Get-LinkItem

if ($Undo) {
    if (Test-IsReparsePoint $item) {
        # Deleting a junction removes only the link, never the files it points to.
        [IO.Directory]::Delete($linkPath)
        Write-Host "Removed the junction at .agents\skills."
    }
    Invoke-Git update-index --no-skip-worktree -- $gitPath
    Invoke-Git checkout -- $gitPath
    Write-Host "Restored .agents\skills as git checked it out."
    exit 0
}

if (Test-IsReparsePoint $item) {
    $current = [string]$item.Target
    Write-Host ".agents\skills is already a link ($($item.LinkType) -> $current). Nothing to do."
    exit 0
}

if ($null -ne $item -and $item.PSIsContainer) {
    throw ".agents\skills is a real folder, not the file git checks out. Move it away and run the script again."
}

if ($null -ne $item) {
    # The text file git writes in place of the symlink; it holds "../.claude/skills".
    Remove-Item -LiteralPath $linkPath -Force
}

$agentsDir = Split-Path -Parent $linkPath
if (-not (Test-Path -LiteralPath $agentsDir)) {
    New-Item -ItemType Directory -Path $agentsDir | Out-Null
}

New-Item -ItemType Junction -Path $linkPath -Target $targetPath | Out-Null
Write-Host "Linked .agents\skills -> $targetPath (junction)."

# Keep the local change out of git status and out of commits.
Invoke-Git update-index --skip-worktree -- $gitPath
Write-Host "git now ignores the local change to .agents\skills (skip-worktree)."

$skillCount = @(Get-ChildItem -LiteralPath $linkPath -Directory).Count
Write-Host "Codex can now see $skillCount skills in .agents\skills."
