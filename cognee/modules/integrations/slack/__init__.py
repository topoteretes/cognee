"""Slack integration package.

Importing this package registers the Slack adapter with the integrations
registry as a side effect — mirrors
:mod:`cognee.modules.tools.builtin`, which registers built-in tools the same
way at import time.
"""

from cognee.modules.integrations.registry import use_integration
from cognee.modules.integrations.slack.adapter import SlackIntegration

use_integration(SlackIntegration())
