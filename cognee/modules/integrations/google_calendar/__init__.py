"""Google Calendar OAuth integration."""

from cognee.modules.integrations.google_calendar.adapter import GoogleCalendarIntegration
from cognee.modules.integrations.registry import use_integration

use_integration(GoogleCalendarIntegration())

__all__ = ["GoogleCalendarIntegration"]
