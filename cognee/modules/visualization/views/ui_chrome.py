"""UI chrome: theme toggle + tab switching (Graph / Schema).

Small, generic UI plumbing that lives outside the canvas renderer's IIFE.
"""

import os

_JS_PATH = os.path.join(os.path.dirname(__file__), "ui_chrome.js")


def emit_js(_preprocessed=None) -> str:
    with open(_JS_PATH, "r", encoding="utf-8") as f:
        return f.read()
