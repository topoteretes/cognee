# File: eval_with_modal.py
from shutil import ignore_patterns

import modal
import os
import json
from pathlib import Path
from typing import Optional
import sys
import dotenv

dotenv.load_dotenv()
app = modal.App("cognee-runner")

# Get the parent directory path
PARENT_DIR = Path(__file__).resolve().parent.parent


MODAL_DOCKERFILE_PATH = Path( "Dockerfile.modal")


# Define ignore patterns
IGNORE_PATTERNS = [
    ".venv/**/*",
    "__pycache__",
    "*.pyc",
    ".git",
    ".pytest_cache",
    "*.egg-info",
    "RAW_GIT_REPOS/**/*"
]

# Create image from Modal-specific Dockerfile
image = modal.Image.from_dockerfile(
    path=MODAL_DOCKERFILE_PATH,
    gpu="T4",
    force_build=False,
    ignore=IGNORE_PATTERNS
).copy_local_file("pyproject.toml", "pyproject.toml").copy_local_file("poetry.lock", "poetry.lock").env({"ENV": os.getenv('ENV'), "LLM_API_KEY": os.getenv("LLM_API_KEY")}).poetry_install_from_file(poetry_pyproject_toml="pyproject.toml")


@app.function(
    image=image,
    gpu="T4",
    concurrency_limit=5,
    timeout=9000000
)
async def run_single_repo(instance_data: dict, disable_cognee: bool = False):
    import os
    import json
    from process_single_repo import process_repo  # Import the async function directly

    # Process the instance
    result = await process_repo(instance_data, disable_cognee=disable_cognee)

    # Save the result
    instance_id = instance_data["instance_id"]
    filename = f"pred_{'nocognee' if disable_cognee else 'cognee'}_{instance_id}.json"
    path_in_container = os.path.join("/app", filename)

    with open(path_in_container, "w") as f:
        json.dump(result, f, indent=2)

    with open(path_in_container, "r") as f:
        content = f.read()

    return (filename, content)
# async def run_single_repo(instance_data: dict, disable_cognee: bool = False):
#     import subprocess
#     import json
#     import os
#
#     # Install project dependencies
#     subprocess.run(
#         ["poetry", "install", "--no-interaction"],
#         cwd="/app",
#         check=True
#     )
#
#     instance_json_str = json.dumps(instance_data)
#
#     # cmd = [
#     #     "poetry",
#     #     "run",
#     #     "python",
#     #     "process_single_repo.py",
#     #     f"--instance_json={instance_json_str}",
#     # ]
#     # if disable_cognee:
#     #     cmd.append("--disable-cognee")
#     #
#     # subprocess.run(cmd, cwd="/app", check=True, env={
#     #             "ENV": os.getenv('ENV'),
#     #             "LLM_API_KEY": os.getenv("LLM_API_KEY")
#     #         })
#     venv_python = os.path.join("venv", "bin", "python")  # Use "Scripts" instead of "bin" on Windows
#
#     cmd = [
#         "poetry",
#         "run",
#         "process_single_repo.py",
#         f"--instance_json={instance_json_str}",
#     ]
#     if disable_cognee:
#         cmd.append("--disable-cognee")
#
#     subprocess.run(cmd, cwd="/app", check=True, env={
#         "ENV": os.getenv('ENV'),
#         "LLM_API_KEY": os.getenv("LLM_API_KEY")
#     })
#     instance_id = instance_data["instance_id"]
#     filename = f"pred_{'nocognee' if disable_cognee else 'cognee'}_{instance_id}.json"
#     path_in_container = os.path.join("/app", filename)
#
#     if os.path.exists(path_in_container):
#         with open(path_in_container, "r") as f:
#             content = f.read()
#         return (filename, content)
#     else:
#         return (filename, "")


@app.local_entrypoint()
async def main(disable_cognee: bool = False, num_samples: int = 2):
    import subprocess
    import json
    import os
    def check_install_package(package_name):
        """Check if a pip package is installed and install it if not."""
        try:
            __import__(package_name)
            return True
        except ImportError:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
                return True
            except subprocess.CalledProcessError:
                return False

    for dependency in ["transformers", "sentencepiece", "swebench","python-dotenv"]:
        check_install_package(dependency)

    from swebench.harness.utils import load_swebench_dataset

    print(f"Configuration:")
    print(f"• Running in Modal mode")
    print(f"• Disable Cognee: {disable_cognee}")
    print(f"• Number of samples: {num_samples}")

    dataset_name = (
        "princeton-nlp/SWE-bench_Lite_bm25_13K" if disable_cognee
        else "princeton-nlp/SWE-bench_Lite"
    )

    swe_dataset = load_swebench_dataset(dataset_name, split="test")
    swe_dataset = swe_dataset[:num_samples]

    print(f"Processing {num_samples} samples from {dataset_name}")
    import pip

    # Install required dependencies
    pip.main(['install', "pydantic>=2.0.0", "pydantic-settings>=2.0.0"])

    tasks = [
        run_single_repo.remote(instance, disable_cognee=disable_cognee)
        for instance in swe_dataset
    ]
    import asyncio
    # Run all tasks concurrently
    results = await asyncio.gather(*tasks)

    # Process results
    merged = []
    for filename, content in results:
        if content:
            with open(filename, "w") as f:
                f.write(content)
            print(f"Saved {filename} locally.")
            merged.append(json.loads(content))

    # Save merged results
    merged_filename = "merged_nocognee.json" if disable_cognee else "merged_cognee.json"
    with open(merged_filename, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Merged {len(merged)} repos into {merged_filename}!")
    print("Done!")

