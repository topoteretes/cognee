import importlib.util
import inspect
from pathlib import Path

import cognee
from cognee.exceptions import CogneeApiError


SAMPLE_ARGUMENTS = {
    "attribute": "sample_attribute",
    "detail": "sample detail",
    "dimension": 1,
    "field": "sample_field",
    "got": "sample",
    "max_index": 2,
    "message": "contract message",
    "name": "ContractError",
    "observer": "sample-observer",
    "provider": "sample-provider",
    "search_type": "sample-search",
    "status_code": 400,
    "value": 1,
}


def _import_exception_modules():
    package_root = Path(cognee.__file__).parent
    for module_file in package_root.rglob("*.py"):
        relative = module_file.relative_to(package_root)
        if "__pycache__" in relative.parts:
            continue
        if module_file.name == "__init__.py":
            continue
        if "exceptions" not in relative.parts and module_file.name != "exceptions.py":
            continue

        module_name = "_cognee_exception_contract_" + "_".join(relative.with_suffix("").parts)
        spec = importlib.util.spec_from_file_location(module_name, module_file)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)


def _iter_cognee_error_classes():
    seen = set()
    stack = list(CogneeApiError.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue

        seen.add(cls)
        yield cls
        stack.extend(cls.__subclasses__())


def _build_kwargs(cls):
    kwargs = {}
    signature = inspect.signature(cls)
    for name, parameter in signature.parameters.items():
        if name == "self":
            continue
        if parameter.default is inspect.Parameter.empty:
            kwargs[name] = SAMPLE_ARGUMENTS[name]
        elif name in {"message", "name", "status_code"}:
            kwargs[name] = SAMPLE_ARGUMENTS[name]

    return kwargs


def test_all_cognee_error_subclasses_preserve_message_args_and_chaining():
    _import_exception_modules()

    classes = list(_iter_cognee_error_classes())
    assert classes

    for cls in classes:
        exc = cls(**_build_kwargs(cls))

        assert exc.message
        assert str(exc)
        assert exc.message in str(exc)
        assert exc.args
        assert exc.args[0] == exc.message

        cause = RuntimeError("root cause")
        try:
            raise exc from cause
        except cls as raised:
            assert raised.__cause__ is cause
