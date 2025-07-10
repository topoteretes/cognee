from cognee.api.v1.add.config import get_s3_config


def get_s3_fs():
    s3_config = get_s3_config()

    fs = None
    if s3_config.aws_access_key_id is not None and s3_config.aws_secret_access_key is not None:
        import s3fs

        fs = s3fs.S3FileSystem(
            key=s3_config.aws_access_key_id, secret=s3_config.aws_secret_access_key, anon=False
        )
    return fs
