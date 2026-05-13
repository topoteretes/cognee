from __future__ import annotations

from math import inf
from typing import Any

from cognee.eval_framework.benchmark_adapters.logistics_system_utils.entities.delivery_models import (
    DeliveryPlan,
    RejectedOption,
    TransportEvaluation,
)
from cognee.eval_framework.benchmark_adapters.logistics_system_utils.entities.enums import (
    DeliveryPriority,
    PackageCategory,
    PackageState,
    Region,
    ShippingRange,
    TransportMode,
)
from cognee.eval_framework.benchmark_adapters.logistics_system_utils.entities.package import Package
from cognee.eval_framework.benchmark_adapters.logistics_system_utils.entities.post_office import (
    PostOffice,
)
from cognee.eval_framework.benchmark_adapters.logistics_system_utils.entities.retailer import (
    Retailer,
)
from cognee.eval_framework.benchmark_adapters.logistics_system_utils.entities.transportation import (
    Carrier,
)
from cognee.eval_framework.benchmark_adapters.logistics_system_utils.entities.user import User


class DeliveryRuleEngine:
    DELIVERY_TIME_RULES = {
        "region_distance": "R10",
        "formula": "R11",
        "weekend_reduction": "R12",
        "tier_bonus": "R13",
        "priority_adjustment": "R14",
        "minimum_bound": "R15",
        "shipping_range_distance_score": "R24",
        "region_distance_score": "R25",
        "carrier_base_delay": "R26",
        "mode_transit_days": "R29",
    }
    PRICE_RULES = {
        "region_distance": "R10",
        "formula": "R16",
        "priority_adjustment": "R40",
        "minimum_bound": "R17",
        "shipping_range_distance_score": "R24",
        "region_distance_score": "R25",
        "carrier_reliability": "R27",
        "mode_cost_score": "R28",
    }
    ROUTE_RULES = {
        "route_participation": "R18",
        "route_order": "R19",
    }
    SELECTION_REASON_RULES = {
        "carrier_range": "R01",
        "weekend_reduction": "R12",
        "temperature_control": "R03",
        "hazardous_certification": "R04",
        "mode_priority_alignment": "R23",
    }

    def _resolve_world_actor(
        self,
        world: dict[str, Any],
        plural_key: str,
        singular_key: str,
    ) -> Any:
        items = world.get(plural_key, [])
        if items:
            return items[0]

        item = world.get(singular_key)
        if item is not None:
            return item

        raise ValueError(f"World does not contain `{plural_key}` or `{singular_key}`.")

    def evaluate(
        self,
        retailer: Retailer,
        user: User,
        world: dict[str, Any],
        package_index: int = 0,
        package: Package = None,
    ) -> DeliveryPlan:
        post_offices = world["post_offices"]
        carriers = world["carriers"]
        packages = world["packages"]

        package = self._resolve_package(
            package_index=package_index, packages=packages, package=package
        )
        post_office_map = {post_office.post_office_id: post_office for post_office in post_offices}

        if package.current_state is PackageState.DELIVERED:
            return DeliveryPlan(
                retailer=retailer,
                user=user,
                package=package,
                selected_option=None,
                rejected_options=[],
                status_message="Package already delivered to the recipient.",
            )

        valid_options: list[TransportEvaluation] = []
        rejected_options: list[RejectedOption] = []

        for carrier in carriers:
            for mode in carrier.supported_modes:
                reasons = self._validation_failures(retailer, user, package, carrier, mode)
                if reasons:
                    rejected_options.append(
                        RejectedOption(
                            carrier_name=carrier.company_name,
                            mode=mode,
                            reasons=reasons,
                        )
                    )
                    continue

                selection_reason_details = self._selection_reason_details(
                    retailer, user, package, carrier, mode
                )
                valid_options.append(
                    TransportEvaluation(
                        carrier_name=carrier.company_name,
                        mode=mode,
                        estimated_delivery_days=self._estimate_delivery_days(
                            retailer, user, package, carrier, mode
                        ),
                        estimated_delivery_days_supporting_facts=self._delivery_days_fact_descriptions(
                            retailer=retailer,
                            user=user,
                            package=package,
                            carrier=carrier,
                            mode=mode,
                        ),
                        estimated_delivery_days_supporting_facts_data_sources=self._delivery_days_supporting_facts(
                            retailer=retailer,
                            user=user,
                            package=package,
                            carrier=carrier,
                            mode=mode,
                        ),
                        estimated_transport_price=self._estimate_transport_price(
                            retailer, user, package, carrier, mode
                        ),
                        estimated_transport_price_supporting_facts=self._transport_price_fact_descriptions(
                            retailer=retailer,
                            user=user,
                            package=package,
                            carrier=carrier,
                            mode=mode,
                        ),
                        estimated_transport_price_supporting_facts_data_sources=self._transport_price_supporting_facts(
                            retailer=retailer,
                            user=user,
                            package=package,
                            carrier=carrier,
                            mode=mode,
                        ),
                        score=self._score_option(retailer, user, package, carrier, mode),
                        route_summary=self._route_summary(retailer, user, package, post_office_map),
                        route_supporting_facts=self._route_fact_descriptions(
                            retailer=retailer,
                            user=user,
                            package=package,
                            post_office_map=post_office_map,
                        ),
                        route_supporting_facts_data_source=self._route_supporting_facts(
                            retailer=retailer,
                            user=user,
                            package=package,
                            post_office_map=post_office_map,
                        ),
                        reasons=[reason for reason, _ in selection_reason_details],
                        selection_reasons_supporting_facts=[
                            facts for _, facts in selection_reason_details
                        ],
                    )
                )

        selected_option = min(valid_options, key=lambda option: option.score, default=None)
        return DeliveryPlan(
            retailer=retailer,
            user=user,
            package=package,
            selected_option=selected_option,
            rejected_options=rejected_options,
            status_message=self._status_message(package, selected_option),
        )

    def compute_outcome(self, world: dict[str, Any], package_index: int = 0) -> float:
        packages = world.get("packages")
        package = self._resolve_package(
            package_index=package_index,
            packages=packages,
        )
        retailer = self._resolve_world_actor(world, "retailers", "retailer")
        user = self._resolve_world_actor(world, "users", "user")
        delivery_plan = self.evaluate(
            retailer=retailer,
            user=user,
            world=world,
            package_index=package_index,
        )
        if delivery_plan.selected_option is None:
            if package.is_received:
                return 0.0
            return inf
        return delivery_plan.selected_option.estimated_delivery_days

    def _resolve_package(
        self,
        package_index: int,
        packages: list[Package] | None,
        package: Package = None,
    ) -> Package:
        if packages is not None:
            if package_index < 0 or package_index >= len(packages):
                raise IndexError(
                    f"Package index {package_index} is out of range for {len(packages)} package(s)."
                )
            return packages[package_index]

        if package is None:
            raise ValueError("A package list or a single package must be provided for evaluation.")

        if package_index != 0:
            raise IndexError("Package index must be 0 when evaluating a single package.")

        return package

    def _validation_failures(
        self,
        retailer: Retailer,
        user: User,
        package: Package,
        carrier: Carrier,
        mode: TransportMode,
    ) -> list[str]:
        failures: list[str] = []

        if not carrier.supports_range(package.shipping_range):
            failures.append("carrier does not serve the package shipping range")
        if package.shipping_range is ShippingRange.DOMESTIC and retailer.region is not user.region:
            failures.append(
                "domestic deliveries require origin and destination to be in the same region"
            )
        if (
            package.shipping_range is ShippingRange.DOMESTIC
            and carrier.region is not retailer.region
        ):
            failures.append(
                "domestic deliveries require a carrier operating in the same region as the route"
            )
        if (
            package.shipping_range is ShippingRange.REGIONAL
            and retailer.region.is_us != user.region.is_us
        ):
            failures.append(
                "regional deliveries require origin and destination to stay within the same network"
            )
        if (
            package.shipping_range is ShippingRange.REGIONAL
            and carrier.region.is_us != retailer.region.is_us
        ):
            failures.append(
                "regional deliveries require a carrier operating within the same network as the route"
            )
        if package.weight_kg > carrier.max_weight_kg:
            failures.append("package exceeds the carrier maximum supported weight")
        if package.weight_kg > mode.max_recommended_weight_kg:
            failures.append("package is too heavy for the selected transport mode")
        if package.requires_temperature_control and not carrier.temperature_controlled:
            failures.append("package requires temperature-controlled transport")
        if package.requires_hazard_certification and not carrier.hazardous_certified:
            failures.append("package requires hazardous-material certification")
        if package.shipping_range is ShippingRange.INTERCONTINENTAL and mode not in {
            TransportMode.FLIGHT,
            TransportMode.SHIPMENT,
        }:
            failures.append("intercontinental deliveries can only use flight or shipment")
        if package.category is PackageCategory.FRAGILE and mode is TransportMode.SHIPMENT:
            failures.append("fragile packages cannot be assigned to sea shipment in this rule set")
        if package.priority is DeliveryPriority.EXPRESS and mode is TransportMode.SHIPMENT:
            failures.append("express packages cannot use the slowest shipment option")

        return failures

    def _estimate_delivery_days(
        self,
        retailer: Retailer,
        user: User,
        package: Package,
        carrier: Carrier,
        mode: TransportMode,
    ) -> int:
        estimated_days = (
            retailer.processing_days
            + package.shipping_range.distance_score
            + self._region_distance(retailer.region, user.region)
            + mode.transit_days
            + carrier.base_delay_days
        )

        if user.weekend_delivery_eligible and carrier.weekend_operations:
            estimated_days -= 1
        estimated_days -= user.tier.priority_bonus

        if package.priority is DeliveryPriority.EXPRESS and mode in {
            TransportMode.FLIGHT,
            TransportMode.TRUCK,
        }:
            estimated_days -= 1
        if package.priority is DeliveryPriority.ECONOMY and mode in {
            TransportMode.SHIPMENT,
            TransportMode.FREIGHT_RAIL,
        }:
            estimated_days += 1

        return max(1, estimated_days)

    def _score_option(
        self,
        retailer: Retailer,
        user: User,
        package: Package,
        carrier: Carrier,
        mode: TransportMode,
    ) -> float:
        estimated_days = self._estimate_delivery_days(retailer, user, package, carrier, mode)
        estimated_price = self._estimate_transport_price(retailer, user, package, carrier, mode)
        speed_weight = package.priority.speed_weight + user.tier.speed_weight
        cost_weight = package.priority.cost_weight
        reliability_bonus = carrier.reliability_score * 0.5
        return (
            (estimated_days * speed_weight)
            + estimated_price
            + (mode.cost_score * cost_weight)
            - reliability_bonus
        )

    def _delivery_days_supporting_facts(
        self,
        retailer: Retailer,
        user: User,
        package: Package,
        carrier: Carrier,
        mode: TransportMode,
    ) -> list[str]:
        facts = [
            retailer.retailer_id,
            user.user_id,
            package.package_id,
            carrier.carrier_id,
            self.DELIVERY_TIME_RULES["region_distance"],
            self.DELIVERY_TIME_RULES["formula"],
            self.DELIVERY_TIME_RULES["minimum_bound"],
            self.DELIVERY_TIME_RULES["shipping_range_distance_score"],
            self.DELIVERY_TIME_RULES["region_distance_score"],
            self.DELIVERY_TIME_RULES["carrier_base_delay"],
            self.DELIVERY_TIME_RULES["mode_transit_days"],
        ]

        if user.weekend_delivery_eligible and carrier.weekend_operations:
            facts.append(self.DELIVERY_TIME_RULES["weekend_reduction"])
        if user.tier.priority_bonus:
            facts.append(self.DELIVERY_TIME_RULES["tier_bonus"])
        if (
            package.priority is DeliveryPriority.EXPRESS
            and mode in {TransportMode.FLIGHT, TransportMode.TRUCK}
        ) or (
            package.priority is DeliveryPriority.ECONOMY
            and mode in {TransportMode.SHIPMENT, TransportMode.FREIGHT_RAIL}
        ):
            facts.append(self.DELIVERY_TIME_RULES["priority_adjustment"])

        return facts

    def _delivery_days_fact_descriptions(
        self,
        retailer: Retailer,
        user: User,
        package: Package,
        carrier: Carrier,
        mode: TransportMode,
    ) -> list[str]:
        region_distance = self._region_distance(retailer.region, user.region)
        facts = [
            f"{retailer.name} needs {retailer.processing_days} processing day(s) before dispatch",
            f"{package.shipping_range.label} shipping adds {package.shipping_range.distance_score} day(s)",
            f"travel from {retailer.region.display_name} to {user.region.display_name} adds {region_distance} day(s)",
            f"{mode.label} adds {mode.transit_days} transit day(s)",
            f"{carrier.company_name} contributes {carrier.base_delay_days} base delay day(s)",
        ]

        raw_estimated_days = (
            retailer.processing_days
            + package.shipping_range.distance_score
            + region_distance
            + mode.transit_days
            + carrier.base_delay_days
        )

        if user.weekend_delivery_eligible and carrier.weekend_operations:
            facts.append("weekend operations reduce the estimate by 1 day")
            raw_estimated_days -= 1
        if user.tier.priority_bonus:
            facts.append(
                f"{user.tier.label} tier reduces the estimate by {user.tier.priority_bonus} day(s)"
            )
            raw_estimated_days -= user.tier.priority_bonus

        if package.priority is DeliveryPriority.EXPRESS and mode in {
            TransportMode.FLIGHT,
            TransportMode.TRUCK,
        }:
            facts.append(
                f"{package.priority.label} priority reduces the estimate by 1 day for {mode.label}"
            )
            raw_estimated_days -= 1
        if package.priority is DeliveryPriority.ECONOMY and mode in {
            TransportMode.SHIPMENT,
            TransportMode.FREIGHT_RAIL,
        }:
            facts.append(f"{package.priority.label} priority adds 1 day for {mode.label}")
            raw_estimated_days += 1

        if raw_estimated_days < 1:
            facts.append("delivery time is bounded to a minimum of 1 day")

        return facts

    def _selection_reason_details(
        self,
        retailer: Retailer,
        user: User,
        package: Package,
        carrier: Carrier,
        mode: TransportMode,
    ) -> list[tuple[str, list[str]]]:
        reasons: list[tuple[str, list[str]]] = [
            (
                f"{retailer.name} ships from {retailer.region.display_name}",
                [retailer.retailer_id],
            ),
            (
                f"{user.name} receives deliveries in {user.region.display_name}",
                [user.user_id],
            ),
            (
                f"{carrier.company_name} serves {package.shipping_range.label} routes",
                [
                    carrier.carrier_id,
                    package.package_id,
                    self.SELECTION_REASON_RULES["carrier_range"],
                ],
            ),
            (
                f"{mode.label} supports a {package.priority.label} delivery profile",
                [
                    package.package_id,
                    self.SELECTION_REASON_RULES["mode_priority_alignment"],
                ],
            ),
            (
                f"package is currently {package.current_state.label}",
                [package.package_id],
            ),
        ]

        if user.weekend_delivery_eligible and carrier.weekend_operations:
            reasons.append(
                (
                    "weekend operations shorten delivery time",
                    [
                        user.user_id,
                        carrier.carrier_id,
                        self.SELECTION_REASON_RULES["weekend_reduction"],
                    ],
                )
            )
        if package.requires_temperature_control and carrier.temperature_controlled:
            reasons.append(
                (
                    "cold-chain support matches the package category",
                    [
                        package.package_id,
                        carrier.carrier_id,
                        self.SELECTION_REASON_RULES["temperature_control"],
                    ],
                )
            )
        if package.requires_hazard_certification and carrier.hazardous_certified:
            reasons.append(
                (
                    "hazardous certification satisfies compliance requirements",
                    [
                        package.package_id,
                        carrier.carrier_id,
                        self.SELECTION_REASON_RULES["hazardous_certification"],
                    ],
                )
            )

        return reasons

    def _estimate_transport_price(
        self,
        retailer: Retailer,
        user: User,
        package: Package,
        carrier: Carrier,
        mode: TransportMode,
    ) -> float:
        region_cost = self._region_distance(retailer.region, user.region) * 3.0
        weight_cost = package.weight_kg * 0.08
        range_cost = package.shipping_range.distance_score * 2.5
        reliability_cost = max(0, 6 - carrier.reliability_score) * 0.5

        estimated_price = (
            retailer.handling_fee
            + region_cost
            + weight_cost
            + range_cost
            + mode.cost_score
            + reliability_cost
        )
        if package.priority is DeliveryPriority.EXPRESS:
            estimated_price += 8.0
        if package.priority is DeliveryPriority.ECONOMY:
            estimated_price -= 1.5

        return round(max(4.0, estimated_price), 2)

    def _transport_price_supporting_facts(
        self,
        retailer: Retailer,
        user: User,
        package: Package,
        carrier: Carrier,
        mode: TransportMode,
    ) -> list[str]:
        facts = [
            retailer.retailer_id,
            user.user_id,
            package.package_id,
            carrier.carrier_id,
            self.PRICE_RULES["region_distance"],
            self.PRICE_RULES["formula"],
            self.PRICE_RULES["minimum_bound"],
            self.PRICE_RULES["shipping_range_distance_score"],
            self.PRICE_RULES["region_distance_score"],
            self.PRICE_RULES["carrier_reliability"],
            self.PRICE_RULES["mode_cost_score"],
        ]
        if package.priority in {DeliveryPriority.EXPRESS, DeliveryPriority.ECONOMY}:
            facts.append(self.PRICE_RULES["priority_adjustment"])
        return facts

    def _transport_price_fact_descriptions(
        self,
        retailer: Retailer,
        user: User,
        package: Package,
        carrier: Carrier,
        mode: TransportMode,
    ) -> list[str]:
        region_distance = self._region_distance(retailer.region, user.region)
        region_cost = region_distance * 3.0
        weight_cost = package.weight_kg * 0.08
        range_cost = package.shipping_range.distance_score * 2.5
        reliability_cost = max(0, 6 - carrier.reliability_score) * 0.5

        facts = [
            f"{retailer.name} adds a handling fee of ${retailer.handling_fee:.2f}",
            f"travel from {retailer.region.display_name} to {user.region.display_name} adds ${region_cost:.2f} in distance cost",
            f"{package.weight_kg:.1f} kg adds ${weight_cost:.2f} in weight cost",
            f"{package.shipping_range.label} shipping adds ${range_cost:.2f} in range cost",
            f"{mode.label} adds ${float(mode.cost_score):.2f} in mode cost",
            f"{carrier.company_name} reliability adds ${reliability_cost:.2f} in reliability cost",
        ]

        if package.priority is DeliveryPriority.EXPRESS:
            facts.append("express priority adds $8.00 to the price")
        if package.priority is DeliveryPriority.ECONOMY:
            facts.append("economy priority reduces the price by $1.50")

        raw_estimated_price = (
            retailer.handling_fee
            + region_cost
            + weight_cost
            + range_cost
            + mode.cost_score
            + reliability_cost
        )
        if package.priority is DeliveryPriority.EXPRESS:
            raw_estimated_price += 8.0
        if package.priority is DeliveryPriority.ECONOMY:
            raw_estimated_price -= 1.5

        if raw_estimated_price < 4.0:
            facts.append("transport price is bounded to a minimum of $4.00")

        return facts

    def _region_distance(self, origin_region: Region, destination_region: Region) -> int:
        if origin_region is destination_region:
            return 0
        network_penalty = 2 if origin_region.is_us != destination_region.is_us else 0
        return (
            abs(origin_region.distance_score - destination_region.distance_score)
            + 1
            + network_penalty
        )

    def _route_summary(
        self,
        retailer: Retailer,
        user: User,
        package: Package,
        post_office_map: dict[str, PostOffice],
    ) -> str:
        route_parts = [retailer.name]
        seen_post_office_ids: set[str] = set()

        if retailer.origin_post_office_id:
            origin_post_office = post_office_map.get(retailer.origin_post_office_id)
            if origin_post_office is not None:
                route_parts.append(origin_post_office.name)
                seen_post_office_ids.add(origin_post_office.post_office_id)

        for post_office_id in package.route_post_office_ids:
            if post_office_id in seen_post_office_ids:
                continue
            post_office = post_office_map.get(post_office_id)
            if post_office is None:
                continue
            route_parts.append(post_office.name)
            seen_post_office_ids.add(post_office.post_office_id)

        route_parts.append(user.name)
        return " -> ".join(route_parts)

    def _route_fact_descriptions(
        self,
        retailer: Retailer,
        user: User,
        package: Package,
        post_office_map: dict[str, PostOffice],
    ) -> list[str]:
        facts = [f"route starts at {retailer.name}"]
        seen_post_office_ids: set[str] = set()

        if retailer.origin_post_office_id:
            origin_post_office = post_office_map.get(retailer.origin_post_office_id)
            if origin_post_office is not None:
                facts.append(
                    f"{origin_post_office.name} is the origin handoff point for the package"
                )
                seen_post_office_ids.add(origin_post_office.post_office_id)

        for post_office_id in package.route_post_office_ids:
            if post_office_id in seen_post_office_ids:
                continue
            post_office = post_office_map.get(post_office_id)
            if post_office is None:
                continue
            facts.append(f"{post_office.name} is included as a route stop")
            seen_post_office_ids.add(post_office.post_office_id)

        facts.append(f"route ends with delivery to {user.name}")
        return facts

    def _route_supporting_facts(
        self,
        retailer: Retailer,
        user: User,
        package: Package,
        post_office_map: dict[str, PostOffice],
    ) -> list[str]:
        facts = [
            retailer.retailer_id,
            user.user_id,
            package.package_id,
            self.ROUTE_RULES["route_participation"],
            self.ROUTE_RULES["route_order"],
        ]
        seen_post_office_ids: set[str] = set()

        if retailer.origin_post_office_id:
            origin_post_office = post_office_map.get(retailer.origin_post_office_id)
            if origin_post_office is not None:
                facts.append(origin_post_office.post_office_id)
                seen_post_office_ids.add(origin_post_office.post_office_id)

        for post_office_id in package.route_post_office_ids:
            if post_office_id in seen_post_office_ids:
                continue
            post_office = post_office_map.get(post_office_id)
            if post_office is None:
                continue
            facts.append(post_office.post_office_id)
            seen_post_office_ids.add(post_office.post_office_id)

        return facts

    def _status_message(self, package: Package, selected_option: TransportEvaluation | None) -> str:
        if package.current_state is PackageState.RETURNED:
            return "Package has been returned and needs a new fulfillment decision."
        if package.at_post_office and package.current_post_office_name is not None:
            return f"Package is currently at post office {package.current_post_office_name}."
        if package.at_sorting_center:
            return "Package is currently at a sorting center awaiting the next leg."
        if package.in_transport:
            return "Package is already moving through the network."
        if package.current_state is PackageState.CREATED and selected_option is not None:
            return "Package is ready to be dispatched with the selected transport option."
        return f"Package is currently {package.current_state.label}."


def compute_outcome(world: dict[str, Any], package_index: int = 0) -> float:
    return DeliveryRuleEngine().compute_outcome(world, package_index=package_index)
