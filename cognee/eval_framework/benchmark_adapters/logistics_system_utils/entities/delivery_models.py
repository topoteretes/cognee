from __future__ import annotations

import json
from dataclasses import dataclass, field

from cognee.eval_framework.benchmark_adapters.logistics_system_utils.entities.enums import (
    TransportMode,
)
from cognee.eval_framework.benchmark_adapters.logistics_system_utils.entities.package import Package
from cognee.eval_framework.benchmark_adapters.logistics_system_utils.entities.retailer import (
    Retailer,
)
from cognee.eval_framework.benchmark_adapters.logistics_system_utils.entities.user import User


@dataclass(slots=True)
class RejectedOption:
    carrier_name: str
    mode: TransportMode
    reasons: list[str]


@dataclass(slots=True)
class TransportEvaluation:
    carrier_name: str
    mode: TransportMode
    estimated_delivery_days: int
    estimated_delivery_days_supporting_facts: list[str]
    estimated_delivery_days_supporting_facts_data_sources: list[str]
    estimated_transport_price: float
    estimated_transport_price_supporting_facts: list[str]
    estimated_transport_price_supporting_facts_data_sources: list[str]
    score: float
    route_summary: str
    route_supporting_facts: list[str]
    route_supporting_facts_data_source: list[str]
    reasons: list[str] = field(default_factory=list)
    selection_reasons_supporting_facts: list[list[str]] = field(default_factory=list)


@dataclass(slots=True)
class DeliveryPlan:
    retailer: Retailer
    user: User
    package: Package
    selected_option: TransportEvaluation | None
    rejected_options: list[RejectedOption]
    status_message: str = ""

    @property
    def is_deliverable(self) -> bool:
        return self.selected_option is not None or self.package.is_received

    @property
    def estimated_delivery_days(self) -> int | None:
        if self.selected_option is None:
            return 0 if self.package.is_received else None
        return self.selected_option.estimated_delivery_days

    @property
    def estimated_transport_price(self) -> float | None:
        if self.selected_option is None:
            return 0.0 if self.package.is_received else None
        return self.selected_option.estimated_transport_price

    @property
    def route_summary(self) -> str | None:
        if self.selected_option is None:
            return None
        return self.selected_option.route_summary

    def _display_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "package_id": self.package.package_id,
            "retailer_name": self.retailer.name,
            "user_name": self.user.name,
        }

        if self.selected_option is None:
            payload.update(
                {
                    "selected_carrier": "",
                    "carrier_selection_reasons": [],
                    "carrier_selection_reasons_data_sources": [],
                    "estimated_delivery_days": 0 if self.package.is_received else None,
                    "estimated_delivery_days_supporting_facts": [],
                    "estimated_delivery_days_supporting_facts_data_sources": [],
                    "estimated_transport_price": 0.0 if self.package.is_received else None,
                    "estimated_transport_price_supporting_facts": [],
                    "estimated_transport_price_supporting_facts_data_sources": [],
                    "route": "",
                    "route_supporting_facts": [],
                    "route_supporting_facts_data_source": [],
                }
            )
            return payload

        payload.update(
            {
                "selected_carrier": self.selected_option.carrier_name,
                "carrier_selection_reasons": list(self.selected_option.reasons),
                "carrier_selection_reasons_data_sources": [
                    source
                    for facts in self.selected_option.selection_reasons_supporting_facts
                    for source in facts
                ],
                "estimated_delivery_days": self.selected_option.estimated_delivery_days,
                "estimated_delivery_days_supporting_facts": list(
                    self.selected_option.estimated_delivery_days_supporting_facts
                ),
                "estimated_delivery_days_supporting_facts_data_sources": list(
                    self.selected_option.estimated_delivery_days_supporting_facts_data_sources
                ),
                "estimated_transport_price": self.selected_option.estimated_transport_price,
                "estimated_transport_price_supporting_facts": list(
                    self.selected_option.estimated_transport_price_supporting_facts
                ),
                "estimated_transport_price_supporting_facts_data_sources": list(
                    self.selected_option.estimated_transport_price_supporting_facts_data_sources
                ),
                "route": self.selected_option.route_summary,
                "route_supporting_facts": list(self.selected_option.route_supporting_facts),
                "route_supporting_facts_data_source": list(
                    self.selected_option.route_supporting_facts_data_source
                ),
            }
        )

        return payload

    def to_dict(self) -> dict[str, object]:
        return self._display_payload()

    def pretty_print(self, as_json: bool = False) -> str:
        if as_json:
            return json.dumps(self.to_dict(), indent=2)

        payload = self._display_payload()
        lines = [
            f"Package: {payload['package_id']}",
        ]

        if not payload.get("selected_carrier"):
            lines.append("Selected Transport: none")
            return "\n".join(lines)

        lines.extend(
            [
                f"Selected Carrier: {payload['selected_carrier']}",
                f"Estimated Delivery Days: {payload['estimated_delivery_days']}",
                f"Estimated Transport Price: ${float(payload['estimated_transport_price']):.2f}",
                f"Route: {payload['route']}",
            ]
        )

        selection_reasons = payload.get("carrier_selection_reasons", [])
        if selection_reasons:
            lines.append("Selection Reasons:")
            lines.extend(f"  - {reason}" for reason in selection_reasons)

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.pretty_print()
