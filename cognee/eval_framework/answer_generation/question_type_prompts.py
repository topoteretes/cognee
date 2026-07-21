from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional

QUESTION_TYPE_DEFAULT_KEY = "DEFAULT"


def get_question_type_prompt(
    prompt_paths: Mapping[str, str],
    question_type: str,
) -> Optional[str]:
    prompt_path = prompt_paths.get(question_type) or prompt_paths.get(QUESTION_TYPE_DEFAULT_KEY)
    if prompt_path is None:
        return None

    return _read_prompt_file(str(prompt_path))


@lru_cache(maxsize=128)
def _read_prompt_file(prompt_path: str) -> str:
    path = Path(prompt_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path

    return path.read_text(encoding="utf-8").strip()
