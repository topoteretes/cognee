import os


async def get_non_py_files(repo_path):
    """Get files that are not .py files and their contents"""
    if not os.path.exists(repo_path):
        return {}

    IGNORED_PATTERNS = {
        ".git",
        "__pycache__",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        "node_modules",
        "*.egg-info",
    }

    ALLOWED_EXTENSIONS = {
        ".txt",
        ".md",
        ".csv",
        ".json",
        ".xml",
        ".yaml",
        ".yml",
        ".html",
        ".css",
        ".js",
        ".ts",
        ".jsx",
        ".tsx",
        ".sql",
        ".log",
        ".ini",
        ".toml",
        ".properties",
        ".sh",
        ".bash",
        ".dockerfile",
        ".gitignore",
        ".gitattributes",
        ".makefile",
        ".pyproject",
        ".requirements",
        ".env",
        ".pdf",
        ".doc",
        ".docx",
        ".dot",
        ".dotx",
        ".rtf",
        ".wps",
        ".wpd",
        ".odt",
        ".ott",
        ".ottx",
        ".txt",
        ".wp",
        ".sdw",
        ".sdx",
        ".docm",
        ".dotm",
        # Additional extensions for other programming languages
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".cs",
        ".go",
        ".php",
        ".rb",
        ".swift",
        ".pl",
        ".lua",
        ".rs",
        ".scala",
        ".kt",
        ".sh",
        ".sql",
        ".v",
        ".asm",
        ".pas",
        ".d",
        ".ml",
        ".clj",
        ".cljs",
        ".erl",
        ".ex",
        ".exs",
        ".f",
        ".fs",
        ".r",
        ".pyi",
        ".pdb",
        ".ipynb",
        ".rmd",
        ".cabal",
        ".hs",
        ".nim",
        ".vhdl",
        ".verilog",
        ".svelte",
        ".html",
        ".css",
        ".scss",
        ".less",
        ".json5",
        ".yaml",
        ".yml",
    }

    def should_process(path):
        _, ext = os.path.splitext(path)
        return ext in ALLOWED_EXTENSIONS and not any(
            pattern in path for pattern in IGNORED_PATTERNS
        )

    non_py_files_paths = [
        os.path.join(root, file)
        for root, _, files in os.walk(repo_path)
        for file in files
        if not file.endswith(".py") and should_process(os.path.join(root, file))
    ]
    return non_py_files_paths
