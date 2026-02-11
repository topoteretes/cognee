import os
import posixpath
from urllib.parse import urlparse, unquote


def get_data_file_path(file_path: str) -> str:
    """Normalize file paths from various URI schemes to filesystem paths.

    Handles file://, s3://, and regular filesystem paths. Decodes
    percent-encoded characters and preserves UNC network paths.
    """
    parsed = urlparse(file_path)

    if parsed.scheme == "file":
        # file:///path/to/file -> /path/to/file
        fs_path = unquote(parsed.path)

        if os.name == "nt" and parsed.netloc:
            # Handle UNC paths (file://server/share/...)
            fs_path = f"//{parsed.netloc}{fs_path}"

        # Normalize the file URI for Windows - handle drive letters correctly
        if os.name == "nt":  # Windows
            # Handle Windows drive letters correctly: /C:/path -> C:/path
            if (
                (fs_path.startswith("/") or fs_path.startswith("\\"))
                and len(fs_path) > 2
                and fs_path[2] == ":"
                and fs_path[1].isalpha()
            ):
                fs_path = fs_path[1:]

        return os.path.normpath(fs_path)

    elif parsed.scheme == "s3":
        # Handle S3 URLs without normalization (which corrupts them)
        if not parsed.path or parsed.path == "/":
            return f"s3://{parsed.netloc}{parsed.path}"

        normalized_path = posixpath.normpath(parsed.path).lstrip("/")

        return f"s3://{parsed.netloc}/{normalized_path}"

    elif parsed.scheme == "":
        # Regular file path - normalize separators
        return os.path.normpath(file_path)

    else:
        # Other schemes (http, etc.) - return as is or handle as needed
        return file_path
