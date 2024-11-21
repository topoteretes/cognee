import argparse
import asyncio
from cognee.tasks.repo_processor.get_local_dependencies import get_local_script_dependencies

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get local script dependencies.")

    # Suggested path: .../cognee/examples/python/simple_example.py
    parser.add_argument("script_path", type=str, help="Absolute path to the Python script file")

    # Suggested path: .../cognee
    parser.add_argument("repo_path", type=str, help="Absolute path to the repository root")

    args = parser.parse_args()

    dependencies = asyncio.run(get_local_script_dependencies(args.script_path, args.repo_path))

    print("Dependencies:")
    for dependency in dependencies:
        print(dependency)
