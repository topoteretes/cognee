"""Google Gmail OAuth integration."""

from cognee.modules.integrations.gmail.adapter import GoogleGmailIntegration
from cognee.modules.integrations.registry import use_integration

use_integration(GoogleGmailIntegration())

__all__ = ["GoogleGmailIntegration"]
