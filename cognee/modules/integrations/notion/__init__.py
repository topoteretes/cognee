"""Notion integration package.

Importing this package registers the Notion adapter with the integrations registry as
a side effect, the same pattern as the Drive and Linear packages. Notion mounts no
routers of its own (its whole surface is the generic
``/api/v1/integrations/notion/*`` routes, and it receives no webhooks), so the API app
imports this package explicitly to trigger registration.
"""

from cognee.modules.integrations.notion.adapter import NotionIntegration
from cognee.modules.integrations.registry import use_integration

use_integration(NotionIntegration())
