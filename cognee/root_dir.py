from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parent


def get_absolute_path(path_from_root: str) -> str:
    absolute_path = ROOT_DIR / path_from_root
    return str(absolute_path.resolve())


def ensure_absolute_path(
    path: str, base_path: Optional[str] = None, allow_relative: bool = False
) -> str:
    """Ensures a path is absolute, optionally converting relative paths.

    Args:
        path: The path to validate/convert
        base_path: Optional base path for relative paths. If None, uses ROOT_DIR
        allow_relative: If False, raises error for relative paths instead of converting

    Returns:
        Absolute path as string

    Raises:
        ValueError: If path is relative and allow_relative is False
    """
    path_obj = Path(path)
    if path_obj.is_absolute():
        return str(path_obj.resolve())

    if not allow_relative:
        raise ValueError(f"Path must be absolute. Got relative path: {path}")

    base = Path(base_path) if base_path else ROOT_DIR
    return str((base / path).resolve())
