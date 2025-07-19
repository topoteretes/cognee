import pathlib
from os import path
from modal import Image
from logging import getLogger

logger = getLogger("modal_image_creation")

image = Image.from_dockerfile(
    path=pathlib.Path(path.join(path.dirname(__file__), "Dockerfile")).resolve(),
    force_build=False,
).add_local_python_source("cognee", "entrypoint")
