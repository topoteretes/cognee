from typing import List, Union, Literal, Optional
from pydantic import BaseModel

class BaseClass(BaseModel):
    id: str
    name: str
    type: Literal["Class"] = "Class"
    description: str
    constructor_parameters: Optional[List[str]]

class Class(BaseModel):
    id: str
    name: str
    type: Literal["Class"] = "Class"
    description: str
    constructor_parameters: Optional[List[str]]
    from_class: Optional[BaseClass]

class ClassInstance(BaseModel):
    id: str
    name: str
    type: Literal["ClassInstance"] = "ClassInstance"
    description: str
    from_class: Class

class Function(BaseModel):
    id: str
    name: str
    type: Literal["Function"] = "Function"
    description: str
    parameters: Optional[List[str]]
    return_type: str
    is_static: Optional[bool] = False

class Variable(BaseModel):
    id: str
    name: str
    type: Literal["Variable"] = "Variable"
    description: str
    is_static: Optional[bool] = False
    default_value: Optional[str]

class Operator(BaseModel):
    id: str
    name: str
    type: Literal["Operator"] = "Operator"
    description: str
    return_type: str

class ExpressionPart(BaseModel):
    id: str
    name: str
    type: Literal["Expression"] = "Expression"
    description: str
    expression: str
    members: List[Union[Variable, Function, Operator]]

class Expression(BaseModel):
    id: str
    name: str
    type: Literal["Expression"] = "Expression"
    description: str
    expression: str
    members: List[Union[Variable, Function, Operator, ExpressionPart]]

class Edge(BaseModel):
    source_node_id: str
    target_node_id: str
    relationship_name: Literal["called in", "stored in", "defined in", "returned by", "instantiated in", "uses", "updates"]

class SourceCodeGraph(BaseModel):
    id: str
    name: str
    description: str
    language: str
    nodes: List[Union[
        Class,
        Function,
        Variable,
        Operator,
        Expression,
        ClassInstance,
    ]]
    edges: List[Edge]
