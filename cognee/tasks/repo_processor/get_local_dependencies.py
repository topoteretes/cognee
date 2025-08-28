import os
import aiofiles
import importlib
from typing import AsyncGenerator, Optional
from uuid import NAMESPACE_OID, uuid5
import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser, Tree
from cognee.shared.logging_utils import get_logger

from cognee.low_level import DataPoint
from cognee.shared.CodeGraphEntities import (
    CodeFile,
    ImportStatement,
    FunctionDefinition,
    ClassDefinition,
)

logger = get_logger()


class FileParser:
    """
    Handles the parsing of files into source code and an abstract syntax tree
    representation. Public methods include:

    - parse_file: Parses a file and returns its source code and syntax tree representation.
    """

    def __init__(self):
        self.parsed_files = {}

    async def parse_file(self, file_path: str) -> tuple[str, Tree]:
        """
        Parse a file and return its source code along with its syntax tree representation.

        If the file has already been parsed, retrieve the result from memory instead of reading
        the file again.

        Parameters:
        -----------

            - file_path (str): The path of the file to parse.

        Returns:
        --------

            - tuple[str, Tree]: A tuple containing the source code of the file and its
              corresponding syntax tree representation.
        """
        PY_LANGUAGE = Language(tspython.language())
        source_code_parser = Parser(PY_LANGUAGE)

        if file_path not in self.parsed_files:
            source_code = await get_source_code(file_path)
            source_code_tree = source_code_parser.parse(bytes(source_code, "utf-8"))
            self.parsed_files[file_path] = (source_code, source_code_tree)

        return self.parsed_files[file_path]


async def get_source_code(file_path: str):
    """
    Read source code from a file asynchronously.

    This function attempts to open a file specified by the given file path, read its
    contents, and return the source code. In case of any errors during the file reading
    process, it logs an error message and returns None.

    Parameters:
    -----------

        - file_path (str): The path to the file from which to read the source code.

    Returns:
    --------

        Returns the contents of the file as a string if successful, or None if an error
        occurs.
    """
    try:
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            source_code = await f.read()
            return source_code
    except Exception as error:
        logger.error(f"Error reading file {file_path}: {str(error)}")
        return None


def resolve_module_path(module_name):
    """
    Find the file path of a module.

    Return the file path of the specified module if found, or return None if the module does
    not exist or cannot be located.

    Parameters:
    -----------

        - module_name: The name of the module whose file path is to be resolved.

    Returns:
    --------

        The file path of the module as a string or None if the module is not found.
    """
    try:
        spec = importlib.util.find_spec(module_name)
        if spec and spec.origin:
            return spec.origin
    except ModuleNotFoundError:
        return None
    return None


def find_function_location(
    module_path: str, function_name: str, parser: FileParser
) -> Optional[tuple[str, str]]:
    """
    Find the location of a function definition in a specified module.

    Parameters:
    -----------

        - module_path (str): The path to the module where the function is defined.
        - function_name (str): The name of the function whose location is to be found.
        - parser (FileParser): An instance of FileParser used to parse the module's source
          code.

    Returns:
    --------

        - Optional[tuple[str, str]]: Returns a tuple containing the module path and the
          start point of the function if found; otherwise, returns None.
    """
    if not module_path or not os.path.exists(module_path):
        return None

    source_code, tree = parser.parse_file(module_path)
    root_node: Node = tree.root_node

    for node in root_node.children:
        if node.type == "function_definition":
            func_name_node = node.child_by_field_name("name")

            if func_name_node and func_name_node.text.decode() == function_name:
                return (module_path, node.start_point)  # (line, column)

    return None


async def get_local_script_dependencies(
    repo_path: str, script_path: str, detailed_extraction: bool = False
) -> CodeFile:
    """
    Retrieve local script dependencies and create a CodeFile object.

    Parameters:
    -----------

        - repo_path (str): The path to the repository that contains the script.
        - script_path (str): The path of the script for which dependencies are being
          extracted.
        - detailed_extraction (bool): A flag indicating whether to perform a detailed
          extraction of code components.

    Returns:
    --------

        - CodeFile: Returns a CodeFile object containing information about the script,
          including its dependencies and definitions.
    """
    code_file_parser = FileParser()
    source_code, source_code_tree = await code_file_parser.parse_file(script_path)

    file_path_relative_to_repo = script_path[len(repo_path) + 1 :]

    if not detailed_extraction:
        code_file_node = CodeFile(
            id=uuid5(NAMESPACE_OID, script_path),
            name=file_path_relative_to_repo,
            source_code=source_code,
            file_path=script_path,
            language="python",
        )
        return code_file_node

    code_file_node = CodeFile(
        id=uuid5(NAMESPACE_OID, script_path),
        name=file_path_relative_to_repo,
        source_code=None,
        file_path=script_path,
        language="python",
    )

    async for part in extract_code_parts(source_code_tree.root_node, script_path=script_path):
        part.file_path = script_path

        if isinstance(part, FunctionDefinition):
            code_file_node.provides_function_definition.append(part)
        if isinstance(part, ClassDefinition):
            code_file_node.provides_class_definition.append(part)
        if isinstance(part, ImportStatement):
            code_file_node.depends_on.append(part)

    return code_file_node


def find_node(nodes: list[Node], condition: callable) -> Node:
    """
    Find and return the first node that satisfies the given condition.

    Iterate through the provided list of nodes and return the first node for which the
    condition callable returns True. If no such node is found, return None.

    Parameters:
    -----------

        - nodes (list[Node]): A list of Node objects to search through.
        - condition (callable): A callable that takes a Node and returns a boolean
          indicating if the node meets specified criteria.

    Returns:
    --------

        - Node: The first Node that matches the condition, or None if no such node exists.
    """
    for node in nodes:
        if condition(node):
            return node

    return None


async def extract_code_parts(
    tree_root: Node, script_path: str, existing_nodes: list[DataPoint] = {}
) -> AsyncGenerator[DataPoint, None]:
    """
    Extract code parts from a given AST node tree asynchronously.

    Iteratively yields DataPoint nodes representing import statements, function definitions,
    and class definitions found in the children of the specified tree root. The function
    checks
    if nodes are already present in the existing_nodes dictionary to prevent duplicates.
    This function has to be used in an asynchronous context, and it requires a valid
    tree_root
    and proper initialization of existing_nodes.

    Parameters:
    -----------

        - tree_root (Node): The root node of the AST tree containing code parts to extract.
        - script_path (str): The file path of the script from which the AST was generated.
        - existing_nodes (list[DataPoint]): A dictionary that holds already extracted
          DataPoint nodes to avoid duplicates. (default {})

    Returns:
    --------

        Yields DataPoint nodes representing imported modules, functions, and classes.
    """
    for child_node in tree_root.children:
        if child_node.type == "import_statement" or child_node.type == "import_from_statement":
            parts = child_node.text.decode("utf-8").split()

            if parts[0] == "import":
                module_name = parts[1]
                function_name = None
            elif parts[0] == "from":
                module_name = parts[1]
                function_name = parts[3]

                if " as " in function_name:
                    function_name = function_name.split(" as ")[0]

            if " as " in module_name:
                module_name = module_name.split(" as ")[0]

            if function_name and "import " + function_name not in existing_nodes:
                import_statement_node = ImportStatement(
                    name=function_name,
                    module=module_name,
                    start_point=child_node.start_point,
                    end_point=child_node.end_point,
                    file_path=script_path,
                    source_code=child_node.text,
                )
                existing_nodes["import " + function_name] = import_statement_node

            if function_name:
                yield existing_nodes["import " + function_name]

            if module_name not in existing_nodes:
                import_statement_node = ImportStatement(
                    name=module_name,
                    module=module_name,
                    start_point=child_node.start_point,
                    end_point=child_node.end_point,
                    file_path=script_path,
                    source_code=child_node.text,
                )
                existing_nodes[module_name] = import_statement_node

            yield existing_nodes[module_name]

        if child_node.type == "function_definition":
            function_node = find_node(child_node.children, lambda node: node.type == "identifier")
            function_node_name = function_node.text

            if function_node_name not in existing_nodes:
                function_definition_node = FunctionDefinition(
                    name=function_node_name,
                    start_point=child_node.start_point,
                    end_point=child_node.end_point,
                    file_path=script_path,
                    source_code=child_node.text,
                )
                existing_nodes[function_node_name] = function_definition_node

            yield existing_nodes[function_node_name]

        if child_node.type == "class_definition":
            class_name_node = find_node(child_node.children, lambda node: node.type == "identifier")
            class_name_node_name = class_name_node.text

            if class_name_node_name not in existing_nodes:
                class_definition_node = ClassDefinition(
                    name=class_name_node_name,
                    start_point=child_node.start_point,
                    end_point=child_node.end_point,
                    file_path=script_path,
                    source_code=child_node.text,
                )
                existing_nodes[class_name_node_name] = class_definition_node

            yield existing_nodes[class_name_node_name]
