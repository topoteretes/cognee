from tree_sitter import Language, Parser, Node
import tree_sitter_c_sharp as tscsharp
from pathlib import Path
from typing import Optional
from cognee.shared.CodeGraphEntities import CodeFile, CodePart, Dependency, DependencyType


class CSharpFileParser:
    """Parse C# files using tree-sitter."""
    
    def __init__(self):
        self.language = Language(tscsharp.language())
        self.parser = Parser(self.language)
    
    def parse_file(self, file_path: Path, file_content: str) -> Optional[Node]:
        """Parse C# file and return AST root node."""
        try:
            tree = self.parser.parse(bytes(file_content, "utf8"))
            return tree.root_node
        except Exception as e:
            print(f"Error parsing C# file {file_path}: {e}")
            return None


def extract_csharp_code_parts(root_node: Node, file_content: str, file_path: Path) -> tuple[list[CodePart], list[Dependency]]:
    """Extract using directives, classes, and methods from C# AST."""
    code_parts = []
    dependencies = []
    
    def traverse(node: Node):
        # Extract using directives (imports)
        if node.type == "using_directive":
            namespace_node = node.child_by_field_name("name")
            if namespace_node:
                namespace = file_content[namespace_node.start_byte:namespace_node.end_byte]
                dep = Dependency(
                    name=namespace,
                    dependency_type=DependencyType.IMPORT,
                    source_file_path=str(file_path),
                )
                dependencies.append(dep)
        
        # Extract class declarations
        elif node.type == "class_declaration":
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
        
        # Extract interface declarations
        elif node.type == "interface_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                interface_name = file_content[name_node.start_byte:name_node.end_byte]
                code_text = file_content[node.start_byte:node.end_byte]
                code_part = CodePart(
                    name=interface_name,
                    code_type="interface",
                    code=code_text,
                    start_line=node.start_point[0],
                    end_line=node.end_point[0],
                )
                code_parts.append(code_part)
        
        # Extract method declarations
        elif node.type == "method_declaration":
            name_node = node.child_by_field_name("name")
            if name_node:
                method_name = file_content[name_node.start_byte:name_node.end_byte]
                code_text = file_content[node.start_byte:node.end_byte]
                code_part = CodePart(
                    name=method_name,
                    code_type="method",
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


def get_csharp_script_dependencies(file_path: Path, file_content: str) -> CodeFile:
    """Extract dependencies and code structure from C# file."""
    parser = CSharpFileParser()
    root_node = parser.parse_file(file_path, file_content)
    
    if root_node is None:
        # Return minimal CodeFile if parsing fails
        return CodeFile(
            file_path=str(file_path),
            file_name=file_path.name,
            code_parts=[],
            dependencies=[],
        )
    
    code_parts, dependencies = extract_csharp_code_parts(root_node, file_content, file_path)
    
    return CodeFile(
        file_path=str(file_path),
        file_name=file_path.name,
        code_parts=code_parts,
        dependencies=dependencies,
    )
