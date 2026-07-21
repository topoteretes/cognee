from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from cognee.api.v1.settings.routers.get_settings_router import (
    SettingsPayloadDTO,
    get_settings_router,
)


@pytest.mark.asyncio
async def test_save_settings_requires_superuser():
    router = get_settings_router()
    save_settings = next(route.endpoint for route in router.routes if "POST" in route.methods)

    with pytest.raises(HTTPException) as exc_info:
        await save_settings(
            new_settings=SettingsPayloadDTO(),
            user=SimpleNamespace(is_superuser=False),
        )

    assert exc_info.value.status_code == 403
    assert exc_info.value.detail == "Only superusers can modify global settings."
