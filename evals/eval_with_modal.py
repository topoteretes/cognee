# File: eval_with_modal.py

import modal
import os
import json
from typing import Optional

app = modal.App("cognee-runner")

# LOCAL_COGNEE_PATH = os.path.dirname(os.path.abspath(__file__))
LOCAL_COGNEE_PATH = "/Users/vasilije/cognee"

image = (
    modal.Image.debian_slim()
    .pip_install("poetry")
    .copy_local_dir(LOCAL_COGNEE_PATH, "/root/cognee")
    .run_commands(
        "cd /root/cognee && poetry install",
    )
)


@app.function(image=image, gpu="T4", concurrency_limit=5)
def run_single_repo(instance_data: dict, disable_cognee: bool = False):
    import subprocess
    import json
    import os

    instance_json_str = json.dumps(instance_data)

    cmd = [
        "python",
        "process_single_repo.py",
        f"--instance_json={instance_json_str}",
    ]
    if disable_cognee:
        cmd.append("--disable-cognee")

    work_dir = "/root/cognee"
    subprocess.run(cmd, cwd=work_dir, check=True)

    instance_id = instance_data["instance_id"]
    filename = f"pred_{'nocognee' if disable_cognee else 'cognee'}_{instance_id}.json"
    path_in_container = os.path.join(work_dir, filename)

    if os.path.exists(path_in_container):
        with open(path_in_container, "r") as f:
            content = f.read()
        return (filename, content)
    else:
        return (filename, "")


@app.local_entrypoint()
def main(disable_cognee: bool = False, num_samples: int = 5):
    """
    Main entry point for Modal.
    Args:
        disable_cognee: If True, runs without Cognee
        num_samples: Number of samples to process
    """
    from swebench.harness.utils import load_swebench_dataset

    dataset_name = (
        "princeton-nlp/SWE-bench_Lite_bm25_13K" if disable_cognee
        else "princeton-nlp/SWE-bench_Lite"
    )

    swe_dataset = load_swebench_dataset(dataset_name, split="test")
    swe_dataset = swe_dataset[:num_samples]

    calls = []
    for instance in swe_dataset:
        calls.append(run_single_repo.remote(instance, disable_cognee=disable_cognee))

    results = []
    for call in calls:
        filename, content = call
        if content:
            with open(filename, "w") as f:
                f.write(content)
            print(f"Saved {filename} locally.")
            results.append(filename)

    merged = []
    for fname in results:
        with open(fname, "r") as f:
            merged.append(json.load(f))

    merged_filename = "merged_nocognee.json" if disable_cognee else "merged_cognee.json"
    with open(merged_filename, "w") as f:
        json.dump(merged, f, indent=2)

    print(f"Merged {len(results)} repos into {merged_filename}!")
    print("Done!")