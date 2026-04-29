from __future__ import annotations

from dataclasses import dataclass

from cognee.eval_framework.benchmark_adapters.logistics_system_utils.entities.enums import (
    Region,
    ShippingRange,
    UserTier,
)


user_possible_values = {
    "weekend_delivery_eligible": [True, False],
    "user_tier": [tier.label for tier in UserTier],
    "default_shipping_range": [shipping_range.label for shipping_range in ShippingRange],
    "region": [region.label for region in Region],
}


@dataclass(slots=True)
class User:
    user_id: str
    name: str
    tier: UserTier
    region: Region
    weekend_delivery_eligible: bool
    default_shipping_range: ShippingRange

    @classmethod
    def from_labels(
        cls,
        user_id: str,
        name: str,
        tier: str,
        region: str,
        weekend_delivery_eligible: bool,
        default_shipping_range: str,
    ) -> "User":
        return cls(
            user_id=user_id,
            name=name,
            tier=UserTier.from_label(tier),
            region=Region.from_label(region),
            weekend_delivery_eligible=weekend_delivery_eligible,
            default_shipping_range=ShippingRange.from_label(default_shipping_range),
        )
