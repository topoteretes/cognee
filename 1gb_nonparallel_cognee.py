import modal
import os
import asyncio
import cognee
from cognee.shared.logging_utils import get_logger, setup_logging, INFO

logger = get_logger()

app = modal.App("1gb_nonparallel_cognee")

image = (
    modal.Image.from_dockerfile(path="Dockerfile_modal", force_build=False)
    .copy_local_file("pyproject.toml", "pyproject.toml")
    .copy_local_file("poetry.lock", "poetry.lock")
    .env(
        {
            "ENV": os.getenv("ENV"),
            "LLM_API_KEY": os.getenv("LLM_API_KEY"),
            "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
            "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
            "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
        }
    )
    .pip_install("protobuf", "h2", "deepeval", "gdown", "plotly")
)


@app.function(image=image, max_containers=1, timeout=86400, min_containers=1)
async def run_cognee_1gb():
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    setup_logging(log_level=INFO)

    await cognee.add("s3://s3-test-laszlo")
    await cognee.cognify()
    return True


@app.local_entrypoint()
async def main():
    modal_tasks = [run_cognee_1gb.remote.aio()]
    await asyncio.gather(*modal_tasks)
