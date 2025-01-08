import os
import shutil

from git import Repo


def download_github_repo(instance, output_dir):
    """
    Downloads a GitHub repository and checks out the specified commit.

    Args:
        instance (dict): Dictionary containing 'repo', 'base_commit', and 'instance_id'.
        output_dir (str): Directory to store the downloaded repositories.

    Returns:
        str: Path to the downloaded repository.
    """
    repo_owner_repo = instance["repo"]
    base_commit = instance["base_commit"]
    instance_id = instance["instance_id"]

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


def node_to_string(node):
    text = node.attributes["text"]
    type = node.attributes["type"]
    return f"Node(id: {node.id}, type: {type}, description: {text})"


def retrieved_edges_to_string(retrieved_edges):
    edge_strings = []
    for edge in retrieved_edges:
        relationship_type = edge.attributes["relationship_type"]
        edge_str = f"{node_to_string(edge.node1)} {relationship_type} {node_to_string(edge.node2)}"
        edge_strings.append(edge_str)
    return "\n".join(edge_strings)
