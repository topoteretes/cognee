import s3fs
from typing import IO, Optional
from cognee.api.v1.add.config import get_s3_config


def open_data_file(
    file_path: str, mode: str = "rb", encoding: Optional[str] = None, **kwargs
) -> IO:
    if file_path.startswith("s3://"):
        s3_config = get_s3_config()
        if s3_config.aws_access_key_id is not None and s3_config.aws_secret_access_key is not None:
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
    else:
        return open(file_path, mode=mode, encoding=encoding, **kwargs)
