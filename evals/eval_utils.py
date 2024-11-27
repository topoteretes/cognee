import os
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from datasets import Dataset
from swebench.inference.make_datasets.create_instance import make_code_text
from swebench.inference.make_datasets.utils import (AutoContextManager,
                                                    ingest_directory_contents)
from tqdm.auto import tqdm
from git import Repo
import shutil

def ingest_files(filenames):
    files_dict = dict()
    for filename in filenames:
        with open(filename) as f:
            content = f.read()
        files_dict[filename] = content
    return files_dict


def ingest_repos(input_instances):
    orig_dir = os.getcwd()
    with TemporaryDirectory(
        dir="/scratch" if os.path.exists("/scratch") else "/tmp"
    ) as root_dir:
        for instance in tqdm(
            input_instances.values(),
            total=len(input_instances),
            desc="Downloading repos on specific commits",
        ):
            try:
                with AutoContextManager(
                    instance, root_dir
                ) as cm:
                    readmes = cm.get_readme_files()
                    instance["readmes"] = ingest_files(readmes)
                    instance["file_contents"] = ingest_directory_contents(
                        cm.repo_path
                    )
            finally:
                # if AutoContextManager fails to exit properly future exits will return the wrong directory
                os.chdir(orig_dir)

    return input_instances


def extract_fields(instance):
    readmes_text = make_code_text(instance["readmes"])
    code_text = make_code_text(
        instance["file_contents"], add_line_numbers=False)

    text_inputs = "\n".join([readmes_text, code_text])
    text_inputs = text_inputs.strip() + "\n\n"
    # text_inputs = code_text
    patch = "\n".join(["<patch>", instance["patch"], "</patch>"])
    return {**instance, "text": text_inputs, "patch": patch}


def create_dataset(input_instances):
    columns = [
        "instance_id",
        "text",
        "repo",
        "base_commit",
        "problem_statement",
        "hints_text",
        "created_at",
        "patch",
        "test_patch",
        "version",
        "FAIL_TO_PASS",
        "PASS_TO_PASS",
        "environment_setup_commit",
    ]

    data_table = {key: list() for key in columns}
    for instance in input_instances.values():
        datum = extract_fields(instance)
        for key in columns:
            data_table[key].append(datum[key] if key in datum else "")
    dataset = Dataset.from_dict(data_table)

    return dataset


def download_instances(
    input_data,
    path=Path("SWE-bench_testsample"),
    verbose=False,
):
    """Downloads code from github.

    Args:
    - input_data: dictionary with unprocessed input instances.
    - verbose: set ContextManager verbose to True
    """
    input_instances = {x["instance_id"]: x for x in input_data}
    input_instances_copy = deepcopy(input_instances)
    input_instances_with_text = ingest_repos(input_instances_copy)
    dataset = create_dataset(input_instances_with_text)
    dataset.save_to_disk(path)
    return dataset


def download_github_repo(instance, output_dir):
    """
    Downloads a GitHub repository and checks out the specified commit.

    Args:
        instance (dict): Dictionary containing 'repo', 'base_commit', and 'instance_id'.
        output_dir (str): Directory to store the downloaded repositories.

    Returns:
        str: Path to the downloaded repository.
    """
    repo_owner_repo = instance['repo']
    base_commit = instance['base_commit']
    instance_id = instance['instance_id']

    repo_url = f"https://github.com/{repo_owner_repo}.git"

    repo_path = os.path.abspath(os.path.join(output_dir, instance_id))

    # Clone repository if it doesn't already exist
    if not os.path.exists(repo_path):
        print(f"Cloning {repo_url} to {repo_path}...")
        Repo.clone_from(repo_url, repo_path)
    else:
        print(f"Repository already exists at {repo_path}.")


    repo = Repo(repo_path)
    repo.git.checkout(base_commit)

    return repo_path


def delete_repo(repo_path):
    """
    Deletes the specified repository directory.

    Args:
        repo_path (str): Path to the repository to delete.

    Returns:
        None
    """
    try:
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)
            print(f"Deleted repository at {repo_path}.")
        else:
            print(f"Repository path {repo_path} does not exist. Nothing to delete.")
    except Exception as e:
        print(f"Error deleting repository at {repo_path}: {e}")