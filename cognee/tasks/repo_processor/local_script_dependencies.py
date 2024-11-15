from typing import List, Dict, Optional
import jedi
import parso
import sys
from pathlib import Path
from parso.tree import BaseNode


def get_code_entities(node: parso.tree.NodeOrLeaf) -> List[Dict[str, any]]:
    """
    Recursively extract code entities using parso.
    """
    code_entity_list = []

    if not hasattr(node, 'children'):
        return code_entity_list

    # Process nodes of type 'name', which correspond to code entities
    name_nodes = (child for child in node.children if child.type == 'name')
    for name_node in name_nodes:
        code_entity = {
            'name': name_node.value,
            'line': name_node.start_pos[0],
            'column': name_node.start_pos[1]
        }
        code_entity_list.append(code_entity)

    # Recursively process child nodes
    for child in node.children:
        code_entity_list.extend(get_code_entities(child))

    return code_entity_list


def update_code_entity(script: jedi.Script, code_entity: Dict[str, any]) -> None:
    """
    Update a code_entity with (full_name, module_name, module_path) using Jedi
    """
    results = script.goto(code_entity["line"], code_entity["column"], follow_imports=True)
    if results:
        code_entity["full_name"] = getattr(results[0], "full_name", None)
        code_entity["module_name"] = getattr(results[0], "module_name", None)
        code_entity["module_path"] = getattr(results[0], "module_path", None)


def get_local_script_dependencies(script_path: str, repo_path: Optional[str] = None) -> List[str]:
    """
    Extract and return a list of unique module paths that the script depends on.
    """
    if repo_path:
        sys.path.insert(0, str(Path(repo_path).resolve()))

    with open(script_path, "r") as file:
        source_code = file.read()

    script = jedi.Script(code=source_code, path=script_path)

    tree = parso.parse(source_code)
    code_entities = get_code_entities(tree)

    for code_entity in code_entities:
        update_code_entity(script, code_entity)

    module_paths = {
        entity.get("module_path")
        for entity in code_entities
        if entity.get("module_path")
    }
    if repo_path:
        repo_path_resolved = str(Path(repo_path).resolve(strict=False))
        module_paths = {path for path in module_paths if str(path).startswith(repo_path_resolved)}

    return sorted(path for path in module_paths if path)

if __name__ == "__main__":
    # Simple execution example, use absolute paths
    script_path = ".../cognee/examples/python/simple_example.py"
    repo_path = ".../cognee"

    dependencies = get_local_script_dependencies(script_path, repo_path)
    print("Dependencies:")
    for dependency in dependencies:
        print(dependency)