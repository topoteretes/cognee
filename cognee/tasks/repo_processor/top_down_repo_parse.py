import os

import jedi
import parso
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)

_NODE_TYPE_MAP = {
    "funcdef": "func_def",
    "classdef": "class_def",
    "async_funcdef": "async_func_def",
    "async_stmt": "async_func_def",
    "simple_stmt": "var_def",
}


def _create_object_dict(name_node, type_name=None):
    return {
        "name": name_node.value,
        "line": name_node.start_pos[0],
        "column": name_node.start_pos[1],
        "type": type_name,
    }


def _parse_node(node):
    """Parse a node to extract importable object details, including async functions and classes."""
    node_type = _NODE_TYPE_MAP.get(node.type)

    if node.type in {"funcdef", "classdef", "async_funcdef"}:
        return [_create_object_dict(node.name, type_name=node_type)]
    if node.type == "async_stmt" and len(node.children) > 1:
        function_node = node.children[1]
        if function_node.type == "funcdef":
            return [
                _create_object_dict(
                    function_node.name, type_name=_NODE_TYPE_MAP.get(function_node.type)
                )
            ]
    if node.type == "simple_stmt":
        # TODO: Handle multi-level/nested unpacking variable definitions in the future
        expr_child = node.children[0]
        if expr_child.type != "expr_stmt":
            return []
        if expr_child.children[0].type == "testlist_star_expr":
            name_targets = expr_child.children[0].children
        else:
            name_targets = expr_child.children
        return [
            _create_object_dict(target, type_name=_NODE_TYPE_MAP.get(target.type))
            for target in name_targets
            if target.type == "name"
        ]
    return []


def extract_importable_objects_with_positions_from_source_code(source_code):
    """Extract top-level objects in a Python source code string with their positions (line/column)."""
    try:
        tree = parso.parse(source_code)
    except Exception as e:
        logger.error(f"Error parsing source code: {e}")
        return []

    importable_objects = []
    try:
        for node in tree.children:
            importable_objects.extend(_parse_node(node))
    except Exception as e:
        logger.error(f"Error extracting nodes from parsed tree: {e}")
        return []

    return importable_objects


def extract_importable_objects_with_positions(file_path):
    """Extract top-level objects in a Python file with their positions (line/column)."""
    try:
        with open(file_path, "r") as file:
            source_code = file.read()
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        return []

    return extract_importable_objects_with_positions_from_source_code(source_code)


def find_entity_usages(script, line, column):
    """
    Return a list of files in the repo where the entity at module_path:line,column is used.
    """
    usages = set()

    try:
        inferred = script.infer(line, column)
    except Exception as e:
        logger.error(f"Error inferring entity at {script.path}:{line},{column}: {e}")
        return []

    if not inferred or not inferred[0]:
        logger.info(f"No entity inferred at {script.path}:{line},{column}")
        return []

    logger.debug(f"Inferred entity: {inferred[0].name}, type: {inferred[0].type}")

    try:
        references = script.get_references(
            line=line, column=column, scope="project", include_builtins=False
        )
    except Exception as e:
        logger.error(
            f"Error retrieving references for entity at {script.path}:{line},{column}: {e}"
        )
        references = []

    for ref in references:
        if ref.module_path:  # Collect unique module paths
            usages.add(ref.module_path)
            logger.info(f"Entity used in: {ref.module_path}")

    return list(usages)


def parse_file_with_references(project, file_path):
    """Parse a file to extract object names and their references within a project."""
    try:
        importable_objects = extract_importable_objects_with_positions(file_path)
    except Exception as e:
        logger.error(f"Error extracting objects from {file_path}: {e}")
        return []

    if not os.path.isfile(file_path):
        logger.warning(f"Module file does not exist: {file_path}")
        return []

    try:
        script = jedi.Script(path=file_path, project=project)
    except Exception as e:
        logger.error(f"Error initializing Jedi Script: {e}")
        return []

    parsed_results = [
        {
            "name": obj["name"],
            "type": obj["type"],
            "references": find_entity_usages(script, obj["line"], obj["column"]),
        }
        for obj in importable_objects
    ]
    return parsed_results


def parse_repo(repo_path):
    """Parse a repository to extract object names, types, and references for all Python files."""
    try:
        project = jedi.Project(path=repo_path)
    except Exception as e:
        logger.error(f"Error creating Jedi project for repository at {repo_path}: {e}")
        return {}

    EXCLUDE_DIRS = {"venv", ".git", "__pycache__", "build"}

    python_files = [
        os.path.join(directory, file)
        for directory, _, filenames in os.walk(repo_path)
        if not any(excluded in directory for excluded in EXCLUDE_DIRS)
        for file in filenames
        if file.endswith(".py") and os.path.getsize(os.path.join(directory, file)) > 0
    ]

    results = {
        file_path: parse_file_with_references(project, file_path)
        for file_path in tqdm(python_files)
    }

    return results
