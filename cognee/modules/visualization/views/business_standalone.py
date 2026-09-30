"""Business view: the UI's canvas renderer, bundled for standalone pages.

``business_standalone.js`` is generated — do not edit it by hand. It is the
esbuild output of ``cognee-frontend/src/standalone/business-standalone.tsx``,
which mounts the same ``BusinessCanvas`` / ``computeBrainState`` code the
Next.js UI uses, with React and d3 bundled in so the page needs no CDN.
Regenerate with ``npm run build:standalone`` in ``cognee-frontend``.
"""

import os

_JS_PATH = os.path.join(os.path.dirname(__file__), "business_standalone.js")


def emit_js(_preprocessed=None) -> str:
    with open(_JS_PATH, "r", encoding="utf-8") as f:
        return f.read()
