from typing import Any, List, Union, Literal, Optional
from cognee.infrastructure.engine import DataPoint

class Variable(DataPoint):
    id: str
    name: str
    type: Literal["Variable"] = "Variable"
    description: str
    is_static: Optional[bool] = False
    default_value: Optional[str] = None
    data_type: str

    _metadata = {
        "index_fields": ["name"]
    }

class Operator(DataPoint):
    id: str
    name: str
    type: Literal["Operator"] = "Operator"
    description: str
    return_type: str

class Class(DataPoint):
    id: str
    name: str
    type: Literal["Class"] = "Class"
    description: str
    constructor_parameters: List[Variable]
    extended_from_class: Optional["Class"] = None
    has_methods: List["Function"]

    _metadata = {
        "index_fields": ["name"]
    }

class ClassInstance(DataPoint):
    id: str
    name: str
    type: Literal["ClassInstance"] = "ClassInstance"
    description: str
    from_class: Class
    instantiated_by: Union["Function"]
    instantiation_arguments: List[Variable]

    _metadata = {
        "index_fields": ["name"]
    }

class Function(DataPoint):
    id: str
    name: str
    type: Literal["Function"] = "Function"
    description: str
    parameters: List[Variable]
    return_type: str
    is_static: Optional[bool] = False

    _metadata = {
        "index_fields": ["name"]
    }

class FunctionCall(DataPoint):
    id: str
    type: Literal["FunctionCall"] = "FunctionCall"
    called_by: Union[Function, Literal["main"]]
    function_called: Function
    function_arguments: List[Any]

class Expression(DataPoint):
    id: str
    name: str
    type: Literal["Expression"] = "Expression"
    description: str
    expression: str
    members: List[Union[Variable, Function, Operator, "Expression"]]

class SourceCodeGraph(DataPoint):
    id: str
    name: str
    description: str
    language: str
    nodes: List[Union[
        Class,
        ClassInstance,
        Function,
        FunctionCall,
        Variable,
        Operator,
        Expression,
    ]]
Class.model_rebuild()
ClassInstance.model_rebuild()
Expression.model_rebuild()
FunctionCall.model_rebuild()
SourceCodeGraph.model_rebuild()
