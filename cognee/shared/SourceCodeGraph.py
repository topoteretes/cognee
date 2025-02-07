from typing import Any, List, Literal, Optional, Union

from cognee.infrastructure.engine import DataPoint


class Variable(DataPoint):
    id: str
    name: str
    description: str
    is_static: Optional[bool] = False
    default_value: Optional[str] = None
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
    constructor_parameters: List[Variable]
    extended_from_class: Optional["Class"] = None
    has_methods: List["Function"]

    metadata: dict = {"index_fields": ["name"]}


class ClassInstance(DataPoint):
    id: str
    name: str
    description: str
    from_class: Class
    instantiated_by: Union["Function"]
    instantiation_arguments: List[Variable]

    metadata: dict = {"index_fields": ["name"]}


class Function(DataPoint):
    id: str
    name: str
    description: str
    parameters: List[Variable]
    return_type: str
    is_static: Optional[bool] = False

    metadata: dict = {"index_fields": ["name"]}


class FunctionCall(DataPoint):
    id: str
    called_by: Union[Function, Literal["main"]]
    function_called: Function
    function_arguments: List[Any]
    metadata: dict = {"index_fields": []}


class Expression(DataPoint):
    id: str
    name: str
    description: str
    expression: str
    members: List[Union[Variable, Function, Operator, "Expression"]]
    metadata: dict = {"index_fields": ["name"]}


class SourceCodeGraph(DataPoint):
    id: str
    name: str
    description: str
    language: str
    nodes: List[
        Union[
            Class,
            ClassInstance,
            Function,
            FunctionCall,
            Variable,
            Operator,
            Expression,
        ]
    ]
    metadata: dict = {"index_fields": ["name"]}


Class.model_rebuild()
ClassInstance.model_rebuild()
Expression.model_rebuild()
FunctionCall.model_rebuild()
SourceCodeGraph.model_rebuild()
