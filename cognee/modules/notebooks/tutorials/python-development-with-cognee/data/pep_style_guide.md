# PEP 8 Style Guide: Essentials

## Code Layout
- Indentation: 4 spaces per level
- Line length: 79 for code (88/100 acceptable by team), 72 for comments/docstrings
- Blank lines: 2 around top-level defs/classes, 1 between methods

```python
# Hanging indent for long calls
foo = long_function_name(
    var_one, var_two,
    var_three, var_four,
)
```

## Imports
- One import per line
- Group: stdlib, third-party, local
- Prefer absolute imports; avoid wildcard imports

```python
import os
import sys
from subprocess import Popen, PIPE

import requests

from myproject.models import User
```

## Whitespace
- No space inside brackets or before commas/semicolons
- Spaces around binary operators

```python
x = 1
hypot2 = x * x + y * y
```

## Naming
- snake_case: functions, variables
- PascalCase: classes
- SCREAMING_SNAKE_CASE: constants

## Comments & Docstrings
- Use complete sentences; keep up to date
- Triple-double quotes for public modules, classes, functions
```python
def f(x: int) -> int:
    """Return x doubled."""
    return x * 2
```

## Type Hints
- Space after colon; arrow for returns
```python
def munge(s: str) -> str: ...
```

## Tooling
- Black, isort, Flake8 (or Ruff) to automate style
- Example pyproject.toml excerpt:
```toml
[tool.black]
line-length = 88

[tool.isort]
profile = "black"
```

## Common Violations
- E501: line too long -> break with parentheses
- E225: missing whitespace around operator
- E402: module import not at top of file
