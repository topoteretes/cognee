"""Google Calendar connector for cognee.

Exports the contract object (``CONNECTOR``), the DLT source factory
(``google_calendar_source``) and the client builders and helpers cognee's connector
package exported. This module imports no host: cognee is needed only by
:mod:`cognee_connector_google_calendar.cognee_host`, which is installed with the
``cognee`` extra and which nothing here imports.
"""

from cognee_connector_google_calendar.connector import (
    CONNECTOR,
    DESCRIPTOR,
    GoogleCalendarConnector,
)
from cognee_connector_google_calendar.source import (
    GoogleCalendarError,
    GoogleCalendarSettings,
    build_calendar_service,
    build_calendar_service_from_access_token,
    google_calendar_source,
    list_calendars,
)

__all__ = [
    "CONNECTOR",
    "DESCRIPTOR",
    "GoogleCalendarConnector",
    "GoogleCalendarError",
    "GoogleCalendarSettings",
    "build_calendar_service",
    "build_calendar_service_from_access_token",
    "google_calendar_source",
    "list_calendars",
]
