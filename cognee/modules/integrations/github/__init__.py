"""GitHub App integration package.

Importing this package registers the GitHub adapter with the integrations
registry as a side effect — the same pattern as the Slack package. Unlike
Slack, GitHub mounts no routers of its own (its webhooks arrive on the
generic ``POST /api/v1/integrations/github/events`` route), so the API app
imports this package explicitly to trigger registration.
"""

from cognee.modules.integrations.github.adapter import GithubIntegration
from cognee.modules.integrations.registry import use_integration

use_integration(GithubIntegration())
