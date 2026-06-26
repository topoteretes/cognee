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

    # resolve(strict=False) canonicalizes symlinks (and e.g. macOS /tmp -> /private/tmp)
    # without requiring the path to exist, so containment checks below are symlink-safe.
    return tuple(Path(os.path.expanduser(root)).resolve(strict=False) for root in root_values)


def _is_path_under_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_local_path(path: str | Path, *, must_exist: bool = False) -> Path:
    # resolve(strict=False) follows symlinks, so a path cannot escape an allowed
    # root through a symlink that points outside it (a lexical abspath check would
    # accept such a path).
    resolved_path = Path(os.path.expanduser(os.fspath(path))).resolve(strict=False)
    for root in get_allowed_local_file_roots():
        if _is_path_under_root(resolved_path, root):
            if must_exist and not resolved_path.exists():
                raise FileNotFoundError(path)
            return resolved_path

    raise ValueError("Local file path is outside allowed roots.")
