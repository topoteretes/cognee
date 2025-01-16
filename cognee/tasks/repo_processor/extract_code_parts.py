from typing import Dict, List
import parso
import logging

logger = logging.getLogger(__name__)


def _extract_parts_from_module(module, parts_dict: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Extract code parts from a parsed module."""

    current_top_level_code = []
    child_to_code_type = {
        "classdef": "classes",
        "funcdef": "functions",
        "import_name": "imports",
        "import_from": "imports",
    }

    for child in module.children:
        if child.type == "simple_stmt":
            current_top_level_code.append(child.get_code())
            continue

        if current_top_level_code:
            parts_dict["top_level_code"].append("\n".join(current_top_level_code))
            current_top_level_code = []

        if child.type in child_to_code_type:
            code_type = child_to_code_type[child.type]
            parts_dict[code_type].append(child.get_code())

    if current_top_level_code:
        parts_dict["top_level_code"].append("\n".join(current_top_level_code))

    if parts_dict["imports"]:
        parts_dict["imports"] = ["\n".join(parts_dict["imports"])]

    return parts_dict


def extract_code_parts(source_code: str) -> Dict[str, List[str]]:
    """Extract high-level parts of the source code."""

    parts_dict = {"classes": [], "functions": [], "imports": [], "top_level_code": []}

    if not source_code.strip():
        logger.warning("Empty source_code provided.")
        return parts_dict

    try:
        module = parso.parse(source_code)
    except Exception as e:
        logger.error(f"Error parsing source code: {e}")
        return parts_dict

    if not module.children:
        logger.warning("Parsed module has no children (empty or invalid source code).")
        return parts_dict

    return _extract_parts_from_module(module, parts_dict)
