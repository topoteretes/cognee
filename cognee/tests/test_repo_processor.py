import os
import shutil
import tempfile
from cognee.tasks.repo_processor.get_repo_file_dependencies import get_source_code_files

def test_get_source_code_files_excludes_common_dirs_and_files():
    # Create a temporary test directory
    test_repo = tempfile.mkdtemp()

    # Create files and folders to include/exclude
    included_file = os.path.join(test_repo, "main.py")
    excluded_dirs = [".venv", "node_modules", "__pycache__", ".git"]
    excluded_files = ["ignore.pyc", "temp.log", "junk.tmp"]

    # Create included file
    with open(included_file, "w") as f:
        f.write("print('Hello world')")

    # Create excluded directories and files inside them
    for folder in excluded_dirs:
        folder_path = os.path.join(test_repo, folder)
        os.makedirs(folder_path)
        file_path = os.path.join(folder_path, "ignored.js")
        with open(file_path, "w") as f:
            f.write("// ignore this")

    # Create excluded files in root
    for file_name in excluded_files:
        file_path = os.path.join(test_repo, file_name)
        with open(file_path, "w") as f:
            f.write("dummy")

    # Run function
    results = get_source_code_files(test_repo)

    # Assert only included file is present
    assert included_file in results
    for root, dirs, files in os.walk(test_repo):
        for name in files:
            full_path = os.path.join(root, name)
            if full_path != included_file:
                assert full_path not in results, f"{full_path} should have been excluded"

    # Cleanup
    shutil.rmtree(test_repo)