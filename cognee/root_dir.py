from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parent


def get_absolute_path(path_from_root: str) -> str:
    absolute_path = ROOT_DIR / path_from_root
    return str(absolute_path.resolve())


def ensure_absolute_path(path: str) -> str:
    """Ensures a path is absolute.

    Args:
        path: The path to validate.

    Returns:
        Absolute path as string
    """
    if path is None:
        raise ValueError("Path cannot be None")

    # Check if it's an S3 URL - S3 URLs are absolute by definition
    if path.startswith("s3://"):
        return path

    path_obj = Path(path).expanduser()
    if path_obj.is_absolute():
        return str(path_obj.resolve())

    raise ValueError(f"Path must be absolute. Got relative path: {path}")
