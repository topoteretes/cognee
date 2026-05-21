from typing import Any, Literal

from cognee.infrastructure.engine import DataPoint


class Variable(DataPoint):
    id: str
    name: str
    description: str
    is_static: bool | None = False
    default_value: str | None = None
    data_type: str

    metadata: dict = {"index_fields": ["name"]}


class Operator(DataPoint):
    id: str
    name: str
    description: str
    return_type: str
    metadata: dict = {"index_fields": ["name"]}


class Class(DataPoint):
    id: str
    name: str
    description: str
    constructor_parameters: list[Variable]
    extended_from_class: "Class | None" = None
    has_methods: list["Function"]

    metadata: dict = {"index_fields": ["name"]}


class ClassInstance(DataPoint):
    id: str
    name: str
    description: str
    from_class: Class
    instantiated_by: "Function"
    instantiation_arguments: list[Variable]

    metadata: dict = {"index_fields": ["name"]}


class Function(DataPoint):
    id: str
    name: str
    description: str
    parameters: list[Variable]
    return_type: str
    is_static: bool | None = False

    metadata: dict = {"index_fields": ["name"]}


class FunctionCall(DataPoint):
    id: str
    called_by: Function | Literal["main"]
    function_called: Function
    function_arguments: list[Any]
    metadata: dict = {"index_fields": []}


class Expression(DataPoint):
    id: str
    name: str
    description: str
    expression: str
    members: list[Variable | Function | Operator | "Expression"]
    metadata: dict = {"index_fields": ["name"]}


class SourceCodeGraph(DataPoint):
    id: str
    name: str
    description: str
    language: str
    nodes: list[Class | ClassInstance | Function | FunctionCall | Variable | Operator | Expression]
    metadata: dict = {"index_fields": ["name"]}


Class.model_rebuild()
ClassInstance.model_rebuild()
Expression.model_rebuild()
FunctionCall.model_rebuild()
SourceCodeGraph.model_rebuild()
