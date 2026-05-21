from __future__ import annotations

from dataclasses import dataclass

from cognee.eval_framework.benchmark_adapters.logistics_system_utils.entities.enums import (
    PostOfficeType,
    Region,
    ShippingRange,
)


post_office_possible_values = {
    "office_type": [office_type.label for office_type in PostOfficeType],
    "shipping_range": [shipping_range.label for shipping_range in ShippingRange],
    "region": [region.label for region in Region],
    "supports_cold_chain": [True, False],
    "supports_hazardous_materials": [True, False],
}


@dataclass(slots=True)
class PostOffice:
    post_office_id: str
    name: str
    office_type: PostOfficeType
    region: Region
    shipping_range: ShippingRange
    supports_cold_chain: bool
    supports_hazardous_materials: bool

    @classmethod
    def from_labels(
        cls,
        post_office_id: str,
        name: str,
        office_type: str,
        region: str,
        shipping_range: str,
        supports_cold_chain: bool,
        supports_hazardous_materials: bool,
    ) -> "PostOffice":
        return cls(
            post_office_id=post_office_id,
            name=name,
            office_type=PostOfficeType.from_label(office_type),
            region=Region.from_label(region),
            shipping_range=ShippingRange.from_label(shipping_range),
            supports_cold_chain=supports_cold_chain,
            supports_hazardous_materials=supports_hazardous_materials,
        )
