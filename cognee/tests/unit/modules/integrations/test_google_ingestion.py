"""Missing source packages must not silently select a different ingestion path."""

from unittest.mock import patch

import pytest

from cognee.modules.integrations.google.ingestion import source_factory


@pytest.mark.parametrize("provider", ["gmail", "google_drive"])
def test_missing_connector_has_an_actionable_install_error(provider):
    with (
        patch(
            "cognee.modules.integrations.google.ingestion.import_module",
            side_effect=ModuleNotFoundError("missing connector"),
        ),
        pytest.raises(RuntimeError, match="requires cognee-community-connector-"),
    ):
        source_factory(provider)
