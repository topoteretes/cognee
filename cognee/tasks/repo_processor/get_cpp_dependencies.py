from tree_sitter import Language, Parser, Node
import tree_sitter_cpp as tscpp
from pathlib import Path
from typing import Optional
from cognee.shared.CodeGraphEntities import CodeFile, CodePart, Dependency, DependencyType


class CppFileParser:
    """Parse C++ files using tree-sitter."""
    
    def __init__(self):
        self.language = Language(tscpp.language())
        self.parser = Parser(self.language)
    
    def parse_file(self, file_path: Path, file_content: str) -> Optional[Node]:
        """Parse C++ file and return AST root node."""
        try:
            tree = self.parser.parse(bytes(file_content, "utf8"))
            return tree.root_node
        except Exception as e:
            print(f"Error parsing C++ file {file_path}: {e}")
            return None


def extract_cpp_code_parts(root_node: Node, file_content: str, file_path: Path) -> tuple[list[CodePart], list[Dependency]]:
    """Extract includes, classes, functions, and namespaces from C++ AST."""
    code_parts = []
    dependencies = []
    
    def traverse(node: Node):
        # Extract preprocessor includes
        if node.type == "preproc_include":
            # Get the path node (e.g., <iostream> or "myheader.h")
            for child in node.children:
                if child.type in ["system_lib_string", "string_literal"]:
                    include_path = file_content[child.start_byte:child.end_byte]
                    # Remove quotes or angle brackets
                    include_path = include_path.strip('"<>')
                    dep = Dependency(
                        name=include_path,
                        dependency_type=DependencyType.IMPORT,
                        source_file_path=str(file_path),
                    )
                    dependencies.append(dep)
        
        # Extract class declarations
        elif node.type == "class_specifier":
            name_node = node.child_by_field_name("name")
            if name_node:
                class_name = file_content[name_node.start_byte:name_node.end_byte]
                code_text = file_content[node.start_byte:node.end_byte]
                code_part = CodePart(
                    name=class_name,
                    code_type="class",
                    code=code_text,
                    start_line=node.start_point[0],
                    end_line=node.end_point[0],
                )
                code_parts.append(code_part)
        
        # Extract struct declarations
        elif node.type == "struct_specifier":
            name_node = node.child_by_field_name("name")
            if name_node:
                struct_name = file_content[name_node.start_byte:name_node.end_byte]
                code_text = file_content[node.start_byte:node.end_byte]
                code_part = CodePart(
                    name=struct_name,
                    code_type="struct",
                    code=code_text,
                    start_line=node.start_point[0],
                    end_line=node.end_point[0],
                )
                code_parts.append(code_part)
        
        # Extract function definitions
        elif node.type == "function_definition":
            declarator = node.child_by_field_name("declarator")
            if declarator:
                # Handle different declarator types
                func_name_node = None
                if declarator.type == "function_declarator":
                    func_name_node = declarator.child_by_field_name("declarator")
                elif declarator.type == "pointer_declarator":
                    # Handle pointer functions
                    inner = declarator.child_by_field_name("declarator")
                    if inner and inner.type == "function_declarator":
                        func_name_node = inner.child_by_field_name("declarator")
                
                if func_name_node:
                    func_name = file_content[func_name_node.start_byte:func_name_node.end_byte]
                    code_text = file_content[node.start_byte:node.end_byte]
                    code_part = CodePart(
                        name=func_name,
                        code_type="function",
                        code=code_text,
                        start_line=node.start_point[0],
                        end_line=node.end_point[0],
                    )
                    code_parts.append(code_part)
        
        # Extract namespace definitions
        elif node.type == "namespace_definition":
            name_node = node.child_by_field_name("name")
            if name_node:
                namespace_name = file_content[name_node.start_byte:name_node.end_byte]
                code_text = file_content[node.start_byte:node.end_byte]
                code_part = CodePart(
                    name=namespace_name,
                    code_type="namespace",
                    code=code_text,
                    start_line=node.start_point[0],
                    end_line=node.end_point[0],
                )
                code_parts.append(code_part)
        
        # Recursively traverse child nodes
        for child in node.children:
            traverse(child)
    
    traverse(root_node)
    return code_parts, dependencies


def get_cpp_script_dependencies(file_path: Path, file_content: str) -> CodeFile:
    """Extract dependencies and code structure from C++ file."""
    parser = CppFileParser()
    root_node = parser.parse_file(file_path, file_content)
    
    if root_node is None:
        # Return minimal CodeFile if parsing fails
        return CodeFile(
            file_path=str(file_path),
            file_name=file_path.name,
            code_parts=[],
            dependencies=[],
        )
    
    code_parts, dependencies = extract_cpp_code_parts(root_node, file_content, file_path)
    
    return CodeFile(
        file_path=str(file_path),
        file_name=file_path.name,
        code_parts=code_parts,
        dependencies=dependencies,
    )
