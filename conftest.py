"""Root conftest — stubs broken optional deps before any cognee import.

The mistralai package installed in some environments is an older version that
does not export ``Mistral``.  ``instructor`` checks for it via
``importlib.util.find_spec``, which requires a proper ``__spec__`` on the
stub module.  This shim runs before any test file is collected, so the stub
is always in place first.
"""

import importlib.util
import sys
import types

if "mistralai" not in sys.modules:
    _spec = importlib.util.spec_from_loader("mistralai", loader=None)
    _stub = types.ModuleType("mistralai")
    _stub.__spec__ = _spec
    _stub.Mistral = object
    sys.modules["mistralai"] = _stub
