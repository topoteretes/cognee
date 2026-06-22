from __future__ import annotations

from dataclasses import dataclass

from cognee.eval_framework.benchmark_adapters.logistics_system_utils.entities.enums import (
    DeliveryPriority,
    PackageCategory,
    PackageState,
    ShippingRange,
)


package_possible_values = {
    "weight_kg": [1.0, 5.0, 25.0, 150.0, 900.0],
    "shipping_range": [shipping_range.label for shipping_range in ShippingRange],
    "category": [category.label for category in PackageCategory],
    "priority": [priority.label for priority in DeliveryPriority],
    "insured": [True, False],
    "current_state": [
        package_state.label
        for package_state in PackageState
        if package_state not in {PackageState.DELIVERED, PackageState.RETURNED}
    ],
    "last_known_location": [
        "customs_checkpoint",
        "destination_hub",
        "delivery_vehicle",
        "recipient_address",
    ],
}


@dataclass(slots=True)
class Package:
    package_id: str
    description: str
    weight_kg: float
    shipping_range: ShippingRange
    category: PackageCategory
    priority: DeliveryPriority
    retailer_id: str | None = None
    retailer_name: str | None = None
    user_id: str | None = None
    user_name: str | None = None
    insured: bool = False
    current_state: PackageState = PackageState.CREATED
    last_known_location: str = "origin_warehouse"
    current_post_office_id: str | None = None
    current_post_office_name: str | None = None
    route_post_office_ids: tuple[str, ...] = ()
    route_post_office_names: tuple[str, ...] = ()
    status_history: tuple[str, ...] = ()

    @property
    def requires_temperature_control(self) -> bool:
        return self.category.requires_temperature_control

    @property
    def requires_hazard_certification(self) -> bool:
        return self.category.requires_hazard_certification

    @property
    def is_fragile(self) -> bool:
        return self.category.is_fragile

    @property
    def is_sent(self) -> bool:
        return self.current_state.has_been_sent

    @property
    def is_received(self) -> bool:
        return self.current_state.is_received_by_customer

    @property
    def in_transport(self) -> bool:
        return self.current_state in {
            PackageState.SENT,
            PackageState.RECEIVED_AT_ORIGIN_HUB,
            PackageState.AT_SORTING_CENTER,
            PackageState.IN_TRANSIT,
            PackageState.OUT_FOR_DELIVERY,
        }

    @property
    def at_sorting_center(self) -> bool:
        return self.current_state.is_at_sorting_center

    @property
    def at_post_office(self) -> bool:
        return self.current_post_office_id is not None

    @classmethod
    def from_labels(
        cls,
        package_id: str,
        description: str,
        weight_kg: float,
        shipping_range: str,
        category: str,
        priority: str,
        retailer_id: str | None = None,
        retailer_name: str | None = None,
        user_id: str | None = None,
        user_name: str | None = None,
        insured: bool = False,
        current_state: str = PackageState.CREATED.label,
        last_known_location: str = "origin_warehouse",
        current_post_office_id: str | None = None,
        current_post_office_name: str | None = None,
        route_post_office_ids: tuple[str, ...] = (),
        route_post_office_names: tuple[str, ...] = (),
        status_history: tuple[str, ...] = (),
    ) -> "Package":
        return cls(
            package_id=package_id,
            description=description,
            weight_kg=weight_kg,
            shipping_range=ShippingRange.from_label(shipping_range),
            category=PackageCategory.from_label(category),
            priority=DeliveryPriority.from_label(priority),
            retailer_id=retailer_id,
            retailer_name=retailer_name,
            user_id=user_id,
            user_name=user_name,
            insured=insured,
            current_state=PackageState.from_label(current_state),
            last_known_location=last_known_location,
            current_post_office_id=current_post_office_id,
            current_post_office_name=current_post_office_name,
            route_post_office_ids=route_post_office_ids,
            route_post_office_names=route_post_office_names,
            status_history=status_history,
        )
