from typing import TYPE_CHECKING

from cognee.infrastructure.files.storage.s3_config import get_s3_config

from .RemoteFileStorage import RemoteFileStorage

# Size of botocore's per-client connection pool. Sized above the pipeline's
# default per-dataset item concurrency (20) so concurrent items are not
# serialized by the pool rather than by the network.
S3_MAX_POOL_CONNECTIONS = 32

if TYPE_CHECKING:
    import s3fs  # ty:ignore[unresolved-import]


class S3FileStorage(RemoteFileStorage):
    """File storage on S3 or an S3-compatible store, through s3fs."""

    scheme = "s3"

    def _create_filesystem(self) -> "s3fs.S3FileSystem":
        try:
            import s3fs  # ty:ignore[unresolved-import]
        except ImportError:
            raise ImportError(
                's3fs is required for S3FileStorage. Install it with: pip install cognee"[aws]"'
            )

        s3_config = get_s3_config()
        # botocore pools 10 connections per client by default. Ingestion runs
        # items concurrently (``data_per_batch``, default 20), so once the event
        # loop is no longer the bottleneck that pool becomes one: requests past
        # the tenth queue behind it.
        client_kwargs = {"region_name": s3_config.aws_region}
        config_kwargs = {"max_pool_connections": S3_MAX_POOL_CONNECTIONS}
        if s3_config.aws_access_key_id is not None and s3_config.aws_secret_access_key is not None:
            return s3fs.S3FileSystem(
                key=s3_config.aws_access_key_id,
                secret=s3_config.aws_secret_access_key,
                token=s3_config.aws_session_token,
                anon=False,
                endpoint_url=s3_config.aws_endpoint_url,
                client_kwargs=client_kwargs,
                config_kwargs=config_kwargs,
            )
        # No credentials provided, let s3fs discover them via the IAM Role/Chain
        return s3fs.S3FileSystem(
            anon=False,
            endpoint_url=s3_config.aws_endpoint_url,
            client_kwargs=client_kwargs,
            config_kwargs=config_kwargs,
        )

    @property
    def s3(self) -> "s3fs.S3FileSystem":
        """The s3fs filesystem; the name callers used before the shared base class."""
        return self.fs

    @s3.setter
    def s3(self, filesystem: "s3fs.S3FileSystem") -> None:
        self.fs = filesystem
