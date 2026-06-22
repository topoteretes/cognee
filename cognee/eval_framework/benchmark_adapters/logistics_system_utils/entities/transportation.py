from __future__ import annotations

from dataclasses import dataclass

from cognee.eval_framework.benchmark_adapters.logistics_system_utils.entities.enums import (
    Region,
    ShippingRange,
    TransportMode,
)


transport_possible_values = {
    "region": [region.label for region in Region],
    "temperature_controlled": [True, False],
    "hazardous_certified": [True, False],
    "weekend_operations": [True, False],
    "base_delay_days": [0, 1, 2],
    "shipping_ranges": [shipping_range.label for shipping_range in ShippingRange],
    "transport_modes": [transport_mode.label for transport_mode in TransportMode],
    "max_weight_kg": [500.0, 3000.0, 10000.0, 25000.0],
    "reliability_score": [2, 3, 4, 5],
}


@dataclass(slots=True)
class Carrier:
    carrier_id: str
    company_name: str
    region: Region
    supported_modes: tuple[TransportMode, ...]
    shipping_ranges: tuple[ShippingRange, ...]
    temperature_controlled: bool
    hazardous_certified: bool
    weekend_operations: bool
    max_weight_kg: float
    reliability_score: int
    base_delay_days: int = 0

    def supports_range(self, shipping_range: ShippingRange) -> bool:
        return shipping_range in self.shipping_ranges

    def supports_mode(self, mode: TransportMode) -> bool:
        return mode in self.supported_modes
