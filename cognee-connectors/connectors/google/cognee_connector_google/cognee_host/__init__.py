"""The cognee-host half of the shared Google OAuth code (cognee >= 1.6.5).

Moved from cognee.modules.integrations.google (cognee dev, 2026-10-26):
``oauth_integration`` (the ``GoogleOAuthIntegration`` base the Drive, Gmail and Calendar
adapters subclass) and ``ingestion`` (the Google-named alias of
``cognee.modules.integrations.ingestion`` that the Cloud pod imports). Google itself is
not a provider in cognee's registry, so there is no ``register()`` here: each service
package registers its own adapter (``cognee_connector_google_drive.cognee_host.register``,
``cognee_connector_gmail.cognee_host.register``,
``cognee_connector_google_calendar.cognee_host.register``).

Everything under ``cognee_host/`` may import cognee; it is installed with the ``cognee``
extra (``pip install "cognee-connector-google[cognee]"``). The package ``__init__`` never
imports this subpackage, and nothing here runs at import.
"""
