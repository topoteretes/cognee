from contextlib import contextmanager
from typing import List, Dict, Optional
import jedi
import parso
import sys
from pathlib import Path
from parso.tree import BaseNode


@contextmanager
def add_sys_path(path):
    original_sys_path = sys.path.copy()
    sys.path.insert(0, path)
    try:
        yield
    finally:
        sys.path = original_sys_path


def _get_code_entities(node: parso.tree.NodeOrLeaf) -> List[Dict[str, any]]:
    """
    Recursively extract code entities using parso.
    """
    code_entity_list = []

    if not hasattr(node, 'children'):
        return code_entity_list

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
        code_entity_list.extend(_get_code_entities(child))

    return code_entity_list


def _update_code_entity(script: jedi.Script, code_entity: Dict[str, any]) -> None:
    """
    Update a code_entity with (full_name, module_name, module_path) using Jedi
    """
    results = script.goto(code_entity["line"], code_entity["column"], follow_imports=True)
    if results:
        code_entity["full_name"] = getattr(results[0], "full_name", None)
        code_entity["module_name"] = getattr(results[0], "module_name", None)
        code_entity["module_path"] = getattr(results[0], "module_path", None)

def _extract_dependencies(script_path: str) -> List[str]:
    try:
        with open(script_path, "r") as file:
            source_code = file.read()
    except IOError as e:
        print(f"Error opening {script_path}: {e}")
        return []

    script = jedi.Script(code=source_code, path=script_path)

    tree = parso.parse(source_code)
    code_entities = _get_code_entities(tree)

    for code_entity in code_entities:
        _update_code_entity(script, code_entity)

    module_paths = {
        entity.get("module_path")
        for entity in code_entities
        if entity.get("module_path")
    }

    return sorted(str(path) for path in module_paths)

def get_local_script_dependencies(script_path: str, repo_path: Optional[str] = None) -> List[str]:
    """
    Extract and return a list of unique module paths that the script depends on.
    """
    if repo_path:
        repo_path_resolved = str(Path(repo_path).resolve())
        with add_sys_path(repo_path_resolved):
            dependencies = _extract_dependencies(script_path)
        dependencies = [path for path in dependencies if path.startswith(repo_path_resolved)]
    else:
        dependencies = _extract_dependencies(script_path)
    return dependencies

if __name__ == "__main__":
    # Simple execution example, use absolute paths
    script_path = ".../cognee/examples/python/simple_example.py"
    repo_path = ".../cognee"

    dependencies = get_local_script_dependencies(script_path, repo_path)
    print("Dependencies:")
    for dependency in dependencies:
        print(dependency)