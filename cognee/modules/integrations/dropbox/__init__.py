"""Dropbox integration package.

Importing this package registers the Dropbox adapter with the integrations registry as a
side effect, the same pattern as the Drive and Notion packages. Dropbox mounts no routers of
its own (its whole surface is the generic ``/api/v1/integrations/dropbox/*`` routes, and it
receives no webhooks), so the API app imports this package explicitly to trigger
registration.
"""

from cognee.modules.integrations.dropbox.adapter import DropboxIntegration
from cognee.modules.integrations.registry import use_integration

use_integration(DropboxIntegration())
