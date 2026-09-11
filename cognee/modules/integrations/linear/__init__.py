"""Linear agent integration package.

Importing this package registers the Linear adapter with the integrations
registry as a side effect — the same pattern as the GitHub package. Linear
mounts no routers of its own (its webhooks, agent session events included,
arrive on the generic ``POST /api/v1/integrations/linear/events`` route), so
the API app imports this package explicitly to trigger registration.
"""

from cognee.modules.integrations.linear.adapter import LinearIntegration
from cognee.modules.integrations.registry import use_integration

use_integration(LinearIntegration())
