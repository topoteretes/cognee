"""End-to-end: cognee keeps its files and embedded databases in Azure Blob Storage.

Run against Azurite in CI (``test_azure_file_storage.yml``), once as is and once with
``FILE_ENCRYPTION_ENABLED=true``. Needs ``STORAGE_BACKEND=azure``, ``STORAGE_BUCKET_NAME``
(the container) and Azure credentials, plus LLM and embedding settings.
"""

import os
import pathlib
from uuid import uuid4

import cognee
from cognee.infrastructure.files.encryption.config import get_file_encryption_config
from cognee.infrastructure.files.encryption.envelope import MAGIC
from cognee.infrastructure.files.storage.azure_config import get_azure_config
from cognee.infrastructure.files.storage.remote_storage import get_remote_file_storage

DATASET = "azure_storage_test"
TEXT = (
    "Azure Blob Storage keeps data as blobs inside containers that belong to a storage "
    "account. Azurite is the local emulator Microsoft ships for testing against it."
)
QUESTION = "What is Azurite?"


async def main():
    container = os.environ["STORAGE_BUCKET_NAME"]
    create_container_if_missing(container)
    root = f"az://{container}/{uuid4()}"
    cognee.config.data_root_directory(f"{root}/data")
    cognee.config.system_root_directory(f"{root}/system")

    await cognee.remember(TEXT, dataset_name=DATASET, self_improvement=False)
    answer = await cognee.recall(QUESTION, datasets=[DATASET])
    assert answer, "recall() found nothing in the memory stored on Azure"

    assert_nothing_went_to_a_local_az_folder()
    await assert_databases_are_in_the_container(f"{root}/system")
    await assert_documents_are_in_the_container(f"{root}/data")


def create_container_if_missing(container: str) -> None:
    """A fresh emulator has no containers; cognee itself never creates one."""
    from adlfs import AzureBlobFileSystem

    filesystem = AzureBlobFileSystem(**get_azure_config().adlfs_kwargs())
    if not filesystem.exists(container):
        filesystem.mkdir(container)


def assert_nothing_went_to_a_local_az_folder() -> None:
    """A URL opened as a local path lands in a folder named "az:" in the working directory.

    Empty ``pipeline_run_locks`` folders are the one exception, as with S3: run ownership
    uses OS file locks, which only work on a local filesystem.
    """
    local_files = [path for path in (pathlib.Path.cwd() / "az:").rglob("*") if path.is_file()]
    assert not local_files, f"Files were written locally instead of to Azure: {local_files}"


async def assert_databases_are_in_the_container(system_root: str) -> None:
    """SQLite and Ladybug are uploaded after the run; LanceDB writes to the container itself."""
    stored = await get_remote_file_storage(system_root).list_files("", recursive=True)

    assert any(path.endswith("cognee_db") for path in stored), f"No relational DB: {stored}"
    assert any(path.endswith(".lbug") for path in stored), f"No Ladybug graph: {stored}"
    assert any(".lance" in path for path in stored), f"No LanceDB table: {stored}"


async def assert_documents_are_in_the_container(data_root: str) -> None:
    """The remembered text is stored under the data root: sealed when encryption is on."""
    storage = get_remote_file_storage(data_root)  # the raw backend, no decryption
    stored = await storage.list_files("", recursive=True)
    assert stored, "Nothing was stored under the Azure data root"

    contents = []
    for path in stored:
        async with storage.open(path, mode="rb") as file:
            contents.append(file.read())

    if get_file_encryption_config().enabled:
        assert all(content.startswith(MAGIC) for content in contents), "A document is not sealed"
    else:
        assert any(TEXT.encode() in content for content in contents), "The text is not stored"


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
