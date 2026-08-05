import json
from datetime import date, datetime, timezone
from decimal import Decimal
from uuid import UUID

import pytest

from cognee.modules.storage.utils import JSONEncoder


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (date(2026, 6, 5), "2026-06-05"),
        (datetime(2026, 6, 5, 14, 30, tzinfo=timezone.utc), "2026-06-05T14:30:00+00:00"),
        (UUID("12345678-1234-5678-1234-567812345678"), "12345678-1234-5678-1234-567812345678"),
        (Decimal("12.5"), 12.5),
    ],
)
def test_json_encoder_serializes_supported_types(value, expected):
    encoded = json.dumps({"value": value}, cls=JSONEncoder)

    assert json.loads(encoded) == {"value": expected}


def test_json_encoder_rejects_unsupported_types():
    with pytest.raises(TypeError):
        json.dumps({"value": object()}, cls=JSONEncoder)
