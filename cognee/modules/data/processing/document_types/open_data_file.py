from typing import IO, Optional
from urllib.parse import urlparse
import os
from cognee.api.v1.add.config import get_s3_config


def open_data_file(
    file_path: str, mode: str = "rb", encoding: Optional[str] = None, **kwargs
) -> IO:
    if file_path.startswith("s3://"):
        s3_config = get_s3_config()
        if s3_config.aws_access_key_id is not None and s3_config.aws_secret_access_key is not None:
            import s3fs

            fs = s3fs.S3FileSystem(
                key=s3_config.aws_access_key_id, secret=s3_config.aws_secret_access_key, anon=False
            )
        else:
            raise ValueError("S3 credentials are not set in the configuration.")

        if "b" in mode:
            f = fs.open(file_path, mode=mode, **kwargs)
            if not hasattr(f, "name") or not f.name:
                f.name = file_path.split("/")[-1]
            return f
        else:
            return fs.open(file_path, mode=mode, encoding=encoding, **kwargs)
    elif file_path.startswith("file://"):
        # Handle local file URLs by properly parsing the URI
        parsed_url = urlparse(file_path)
        # On Windows, urlparse handles drive letters correctly
        # Convert the path component to a proper file path
        if os.name == 'nt':  # Windows
            # Remove leading slash from Windows paths like /C:/Users/...
            local_path = parsed_url.path.lstrip('/')
        else:  # Unix-like systems
            local_path = parsed_url.path
        
        return open(local_path, mode=mode, encoding=encoding, **kwargs)
    else:
        return open(file_path, mode=mode, encoding=encoding, **kwargs)
