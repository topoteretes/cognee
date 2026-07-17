"""Test bootstrap for the Slack + cognee bot example.

Puts the example app root on ``sys.path`` so tests can ``import src.<module>``
without installing the package, mirroring ``cognee-mcp/tests`` conventions.
"""

import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]  # examples/slack_cognee_bot/
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
