import modal
import os
import logging
import asyncio
import cognee
import signal
import json
from dotenv import dotenv_values

from cognee.shared.utils import setup_logging
from cognee.modules.search.types import SearchType

logger = logging.getLogger("MODAL_DEPLOYED_INSTANCE")

app = modal.App("cognee-runner")

local_env_vars = dict(dotenv_values(".env"))
print("Modal deployment started with the following environmental variables:")
print(json.dumps(local_env_vars, indent=4))

image = (
    modal.Image.from_dockerfile(path="Dockerfile_modal", force_build=False)
    .add_local_file("pyproject.toml", remote_path="/root/pyproject.toml", copy=True)
    .add_local_file("poetry.lock", remote_path="/root/poetry.lock", copy=True)
    .env(local_env_vars)
    .poetry_install_from_file(poetry_pyproject_toml="pyproject.toml")
    .pip_install("protobuf", "h2")
    .add_local_python_source("cognee")
)


@app.function(image=image, max_containers=5)
async def entry(name: str, text: str):
    setup_logging(logging.INFO)
    logger.info(f"file_name: {name}")
    await cognee.add(text)


def batch_files(file_list, batch_size):
    for i in range(0, len(file_list), batch_size):
        yield file_list[i : i + batch_size]


@app.local_entrypoint()
async def main():
    directory_name = "cognee_parallel_deployment/modal_input/"
    batch_size = 10

    files = [
        os.path.join(directory_name, f)
        for f in os.listdir(directory_name)
        if os.path.isfile(os.path.join(directory_name, f))
    ]

    for batch in batch_files(files, batch_size):
        print("Processing batch:")
        files = []
        for file_path in batch:
            with open(file_path, "r") as file:
                content = file.read()
                # Process the content as needed.
                print(f"Read data from {file_path}")

                files.append({"name": file_path, "text": content})
        print("Batch reading finished...")
        tasks = [entry.remote.aio(item["name"], item["text"]) for item in files]
        await asyncio.gather(*tasks)

    os.kill(os.getpid(), signal.SIGTERM)
