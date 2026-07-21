# Lazy registry: maps harness name → (module_path, class_name)
# The module is only imported when the installer is actually used,
# so platform-specific imports (e.g. winreg) don't break other platforms.

REGISTRY: dict[str, tuple[str, str]] = {
    "claude-code": (
        "cognee.cli.commands.installers.claude_code",
        "ClaudeCodeInstaller",
    ),
    "cursor": (
        "cognee.cli.commands.installers.cursor",
        "CursorInstaller",
    ),
    "opencode": (
        "cognee.cli.commands.installers.opencode",
        "OpenCodeInstaller",
    ),
}


def get_installer(harness: str):
    """Return an instantiated HarnessInstaller for *harness*, or raise KeyError."""
    if harness not in REGISTRY:
        raise KeyError(f"Unknown harness {harness!r}. Supported: {', '.join(sorted(REGISTRY))}")
    module_path, class_name = REGISTRY[harness]
    module = __import__(module_path, fromlist=[class_name])
    cls = getattr(module, class_name)
    return cls()
