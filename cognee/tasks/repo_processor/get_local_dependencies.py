from typing import AsyncGenerator
from uuid import NAMESPACE_OID, uuid5
import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

import aiofiles

import logging

from cognee.low_level import DataPoint
from cognee.shared.CodeGraphEntities import (
    CodeFile,
    ImportStatement,
    FunctionDefinition,
    ClassDefinition,
)

logger = logging.getLogger(__name__)

PY_LANGUAGE = Language(tspython.language())
source_code_parser = Parser(PY_LANGUAGE)


async def get_source_code(file_path: str):
    try:
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            source_code = await f.read()
            return source_code
    except Exception as error:
        logger.error(f"Error reading file {file_path}: {str(error)}")
        return None


async def get_local_script_dependencies(
    repo_path: str, script_path: str, detailed_extraction: bool = False
) -> CodeFile:
    source_code = await get_source_code(script_path)

    relative_file_path = script_path[len(repo_path) + 1 :]

    if not detailed_extraction:
        code_file_node = CodeFile(
            id=uuid5(NAMESPACE_OID, script_path),
            source_code=source_code,
            file_path=relative_file_path,
        )
        return code_file_node

    code_file_node = CodeFile(
        id=uuid5(NAMESPACE_OID, script_path),
        source_code=None,
        file_path=relative_file_path,
    )

    source_code_tree = source_code_parser.parse(bytes(source_code, "utf-8"))

    async for part in extract_code_parts(source_code_tree.root_node):
        part.file_path = relative_file_path

        if isinstance(part, FunctionDefinition):
            code_file_node.provides_function_definition.append(part)
        if isinstance(part, ClassDefinition):
            code_file_node.provides_class_definition.append(part)
        if isinstance(part, ImportStatement):
            code_file_node.depends_on.append(part)

    return code_file_node


def find_node(nodes: list[Node], condition: callable) -> Node:
    for node in nodes:
        if condition(node):
            return node

    return None


async def extract_code_parts(tree_root: Node) -> AsyncGenerator[DataPoint, None]:
    for child_node in tree_root.children:
        if child_node.type == "import_statement":
            module_node = child_node.children[1]
            yield ImportStatement(
                name=module_node.text,
                start_point=child_node.start_point,
                end_point=child_node.end_point,
                source_code=child_node.text,
            )

        if child_node.type == "import_from_statement":
            module_node = child_node.children[1]
            yield ImportStatement(
                name=module_node.text,
                start_point=child_node.start_point,
                end_point=child_node.end_point,
                source_code=child_node.text,
            )

        if child_node.type == "function_definition":
            function_name_node = find_node(
                child_node.children, lambda node: node.type == "identifier"
            )
            yield FunctionDefinition(
                name=function_name_node.text,
                start_point=child_node.start_point,
                end_point=child_node.end_point,
                source_code=child_node.text,
            )

        if child_node.type == "class_definition":
            class_name_node = find_node(child_node.children, lambda node: node.type == "identifier")
            yield ClassDefinition(
                name=class_name_node.text,
                start_point=child_node.start_point,
                end_point=child_node.end_point,
                source_code=child_node.text,
            )
