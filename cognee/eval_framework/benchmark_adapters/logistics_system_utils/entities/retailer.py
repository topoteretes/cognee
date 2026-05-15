from __future__ import annotations

from dataclasses import dataclass

from cognee.eval_framework.benchmark_adapters.logistics_system_utils.entities.enums import (
    Region,
    ShippingRange,
)


retailer_possible_values = {
    "region": [region.label for region in Region],
    "shipping_range": [shipping_range.label for shipping_range in ShippingRange],
    "handling_fee": [2.5, 4.0, 6.5, 9.0],
    "processing_days": [0, 1, 2],
}


@dataclass(slots=True)
class Retailer:
    retailer_id: str
    name: str
    region: Region
    shipping_range: ShippingRange
    handling_fee: float
    processing_days: int
    origin_post_office_id: str | None = None

    @classmethod
    def from_labels(
        cls,
        retailer_id: str,
        name: str,
        region: str,
        shipping_range: str,
        handling_fee: float,
        processing_days: int,
        origin_post_office_id: str | None = None,
    ) -> "Retailer":
        return cls(
            retailer_id=retailer_id,
            name=name,
            region=Region.from_label(region),
            shipping_range=ShippingRange.from_label(shipping_range),
            handling_fee=handling_fee,
            processing_days=processing_days,
            origin_post_office_id=origin_post_office_id,
        )
