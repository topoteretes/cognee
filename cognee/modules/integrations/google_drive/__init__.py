"""Google Drive integration package.

Importing this package registers the Drive adapter with the integrations
registry as a side effect — the same pattern as the GitHub and Linear
packages. Drive mounts no routers of its own (its whole surface is the
generic ``/api/v1/integrations/google_drive/*`` routes, and it receives no
webhooks), so the API app imports this package explicitly to trigger
registration.
"""

from cognee.modules.integrations.google_drive.adapter import GoogleDriveIntegration
from cognee.modules.integrations.registry import use_integration

use_integration(GoogleDriveIntegration())
