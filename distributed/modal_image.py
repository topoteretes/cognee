import json
import pathlib
from os import path
from modal import Image
from logging import getLogger
from dotenv import dotenv_values

logger = getLogger("modal_image_creation")

local_env_vars = dict(dotenv_values(".env"))

logger.debug("Modal deployment started with the following environmental variables:")
logger.debug(json.dumps(local_env_vars, indent=4))

image = (
    Image.from_dockerfile(
        path=pathlib.Path(path.join(path.dirname(__file__), "Dockerfile")).resolve(),
        force_build=False,
    )
    .env(local_env_vars)
    .add_local_python_source("cognee", "entrypoint")
)
