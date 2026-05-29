import os
import tempfile
from pathlib import Path

from cognee.base_config import get_base_config


ALLOWED_LOCAL_FILE_ROOTS_ENV = "COGNEE_ALLOWED_LOCAL_FILE_ROOTS"


def get_allowed_local_file_roots() -> tuple[Path, ...]:
    configured_roots = os.getenv(ALLOWED_LOCAL_FILE_ROOTS_ENV)
    if configured_roots:
        root_values = [root for root in configured_roots.split(os.pathsep) if root]
    else:
        root_values = [str(Path.cwd()), tempfile.gettempdir()]
        base_config = get_base_config()
        root_values.extend(
            [
                base_config.data_root_directory,
                base_config.system_root_directory,
                base_config.cache_root_directory,
                base_config.logs_root_directory,
            ]
        )

    return tuple(
        Path(os.path.normpath(os.path.abspath(os.path.expanduser(root)))) for root in root_values
    )


def _is_path_under_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_local_path(path: str | Path, *, must_exist: bool = False) -> Path:
    resolved_path = os.path.normpath(os.path.abspath(os.path.expanduser(os.fspath(path))))
    for root in get_allowed_local_file_roots():
        root_path = os.fspath(root)

        if resolved_path == root_path:
            if must_exist and not os.path.exists(root_path):
                raise FileNotFoundError(path)
            return Path(root_path)

        root_prefix = root_path if root_path.endswith(os.sep) else f"{root_path}{os.sep}"
        if resolved_path.startswith(root_prefix):
            if must_exist and not os.path.exists(resolved_path):
                raise FileNotFoundError(path)
            return Path(resolved_path)

    raise ValueError("Local file path is outside allowed roots.")
