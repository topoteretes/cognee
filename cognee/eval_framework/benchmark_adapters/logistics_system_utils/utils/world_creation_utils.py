import random

from cognee.eval_framework.benchmark_adapters.logistics_system_utils.entities.enums import (
    DeliveryPriority,
    PackageCategory,
    PackageState,
    PostOfficeType,
    Region,
    ShippingRange,
    TransportMode,
)
from cognee.eval_framework.benchmark_adapters.logistics_system_utils.entities.package import (
    Package,
    package_possible_values,
)
from cognee.eval_framework.benchmark_adapters.logistics_system_utils.entities.post_office import (
    PostOffice,
)
from cognee.eval_framework.benchmark_adapters.logistics_system_utils.entities.retailer import (
    Retailer,
    retailer_possible_values,
)
from cognee.eval_framework.benchmark_adapters.logistics_system_utils.entities.transportation import (
    Carrier,
    transport_possible_values,
)
from cognee.eval_framework.benchmark_adapters.logistics_system_utils.entities.user import (
    User,
    user_possible_values,
)

USER_NAMES = (
    "Ava Chen",
    "Jonah Patel",
    "Sofia Novak",
    "Noah Fischer",
    "Maya West",
    "Lucas Meyer",
    "Elena Rossi",
    "Daniel Brooks",
    "Priya Nair",
    "Ethan Cole",
    "Hana Suzuki",
    "Mateo Alvarez",
    "Lea Schneider",
    "Omar Hassan",
    "Chloe Martin",
)

RETAILER_NAMES = (
    "Northwind Retail",
    "Mercury Market",
    "Harbor Goods",
    "Urban Cart",
    "Summit Supply",
    "Coastal Commerce",
    "Pioneer Fulfillment",
    "Atlas Bazaar",
    "Vertex Mercantile",
    "Maple Street Supply",
)

PACKAGE_DESCRIPTIONS = (
    "consumer electronics",
    "retail replenishment",
    "medical samples",
    "industrial spare parts",
    "kitchen appliances",
    "winter apparel",
    "laboratory supplies",
    "home office equipment",
)

CARRIER_NAMES_BY_REGION = {
    Region.US_NORTH: (
        "Great Lakes Logistics",
        "NorthStar Freight",
    ),
    Region.US_SOUTH: (
        "Sunbelt Cargo",
        "Magnolia Transport",
    ),
    Region.US_EAST: (
        "Atlantic Route Logistics",
        "HarborLine Freight",
    ),
    Region.US_WEST: (
        "Pacific Horizon Logistics",
        "Sierra Freight Lines",
    ),
    Region.US_CENTRAL: (
        "Heartland Shipping",
        "Prairie State Logistics",
    ),
    Region.GERMANY: (
        "Rhein Cargo",
        "Bavaria Freight",
    ),
    Region.FRANCE: (
        "Hexagon Transit",
        "Lyon Freight Solutions",
    ),
    Region.NETHERLANDS: (
        "DeltaLink Logistics",
        "Tulip Freight Services",
    ),
    Region.SPAIN: (
        "Iberia Cargo Partners",
        "Sol Freight",
    ),
    Region.ITALY: (
        "Lombardy Logistics",
        "Italia Cargo Line",
    ),
    Region.POLAND: (
        "Vistula Freight",
        "Mazovia Logistics",
    ),
}


def _pick_unique_name(base_names: tuple[str, ...], used_names: set[str], prefix: str) -> str:
    available_names = [name for name in base_names if name not in used_names]
    if available_names:
        selected_name = random.choice(available_names)
    else:
        selected_name = f"{prefix} {len(used_names) + 1}"
    used_names.add(selected_name)
    return selected_name


def _normalize_entity_count(count: int) -> int:
    return max(2, count)


def _normalize_package_count(count: int) -> int:
    return max(1, count)


def _carrier_name_for_region(region: Region, carrier_index: int) -> str:
    region_names = CARRIER_NAMES_BY_REGION.get(region, ())
    if carrier_index < len(region_names):
        return region_names[carrier_index]
    return f"{region.display_name} Logistics {carrier_index + 1}"


def _secondary_region_for(region: Region) -> Region:
    if region is Region.US_EAST:
        return Region.US_NORTH
    if region is Region.US_WEST:
        return Region.US_CENTRAL
    if region is Region.US_CENTRAL:
        return Region.US_EAST
    if region is Region.US_NORTH:
        return Region.US_EAST
    if region is Region.US_SOUTH:
        return Region.US_CENTRAL
    return region


def _create_post_office(
    name: str,
    office_type: PostOfficeType,
    region: Region,
    shipping_range: ShippingRange,
    supports_cold_chain: bool,
    supports_hazardous_materials: bool,
) -> PostOffice:
    return PostOffice(
        post_office_id=f"po-{random.randint(1000, 9999)}",
        name=name,
        office_type=office_type,
        region=region,
        shipping_range=shipping_range,
        supports_cold_chain=supports_cold_chain,
        supports_hazardous_materials=supports_hazardous_materials,
    )


def _post_offices_by_type(
    post_offices: list[PostOffice],
) -> dict[PostOfficeType, list[PostOffice]]:
    grouped: dict[PostOfficeType, list[PostOffice]] = {
        office_type: [] for office_type in PostOfficeType
    }
    for post_office in post_offices:
        grouped[post_office.office_type].append(post_office)
    return grouped


def _country_key_for_region(region: Region) -> str:
    return "us" if region.is_us else region.label


def _post_offices_in_origin_country(
    post_offices: list[PostOffice], origin_region: Region | None
) -> list[PostOffice]:
    if origin_region is None:
        return post_offices

    origin_country = _country_key_for_region(origin_region)
    filtered_post_offices = [
        post_office
        for post_office in post_offices
        if _country_key_for_region(post_office.region) == origin_country
    ]
    return filtered_post_offices or post_offices


def _origin_post_office_id_for_retailer(
    retailer_region: Region,
    post_offices: list[PostOffice],
) -> str | None:
    country_post_offices = _post_offices_in_origin_country(post_offices, retailer_region)
    prioritized_post_offices = sorted(
        country_post_offices,
        key=lambda post_office: (
            0
            if post_office.office_type is PostOfficeType.ORIGIN_WAREHOUSE
            else 1
            if post_office.office_type is PostOfficeType.ORIGIN_HUB
            else 2,
            0 if post_office.region is retailer_region else 1,
            post_office.name,
        ),
    )
    return prioritized_post_offices[0].post_office_id if prioritized_post_offices else None


def _route_post_offices_for_state(
    package_state: PackageState,
    post_offices: list[PostOffice],
    origin_region: Region | None = None,
    preferred_origin_post_office_id: str | None = None,
) -> list[PostOffice]:
    relevant_post_offices = _post_offices_in_origin_country(post_offices, origin_region)
    grouped = _post_offices_by_type(relevant_post_offices)
    warehouses = grouped[PostOfficeType.ORIGIN_WAREHOUSE]
    hubs = grouped[PostOfficeType.ORIGIN_HUB]
    sorting_centers = grouped[PostOfficeType.SORTING_CENTER]

    preferred_warehouse = next(
        (
            warehouse
            for warehouse in warehouses
            if warehouse.post_office_id == preferred_origin_post_office_id
        ),
        None,
    )
    primary_warehouse = preferred_warehouse or (warehouses[0] if warehouses else None)
    secondary_warehouse = next(
        (
            warehouse
            for warehouse in warehouses
            if warehouse.post_office_id != getattr(primary_warehouse, "post_office_id", None)
        ),
        primary_warehouse,
    )
    primary_hub = (
        hubs[0]
        if hubs
        else (primary_warehouse or (sorting_centers[0] if sorting_centers else None))
    )
    primary_sorting_center = (
        sorting_centers[0] if sorting_centers else (primary_hub or primary_warehouse)
    )
    secondary_sorting_center = (
        sorting_centers[1] if len(sorting_centers) > 1 else primary_sorting_center
    )
    starting_office = primary_warehouse or primary_hub or primary_sorting_center

    state_routes = {
        PackageState.CREATED: [starting_office] if starting_office else [],
        PackageState.SENT: [starting_office] if starting_office else [],
        PackageState.RECEIVED_AT_ORIGIN_HUB: [
            post_office for post_office in (starting_office, primary_hub) if post_office is not None
        ],
        PackageState.AT_SORTING_CENTER: [
            post_office
            for post_office in (starting_office, primary_hub, primary_sorting_center)
            if post_office is not None
        ],
        PackageState.IN_TRANSIT: [
            post_office
            for post_office in (
                starting_office,
                primary_hub,
                primary_sorting_center,
                secondary_sorting_center,
            )
            if post_office is not None
        ],
        PackageState.OUT_FOR_DELIVERY: [
            post_office
            for post_office in (
                starting_office,
                primary_hub,
                primary_sorting_center,
                secondary_sorting_center,
            )
            if post_office is not None
        ],
        PackageState.DELIVERED: [
            post_office
            for post_office in (
                starting_office,
                primary_hub,
                primary_sorting_center,
                secondary_sorting_center,
            )
            if post_office is not None
        ],
        PackageState.RETURNED: [
            post_office
            for post_office in (
                starting_office,
                primary_hub,
                secondary_warehouse or starting_office,
            )
            if post_office is not None
        ],
    }

    return state_routes.get(package_state, [])


def _location_for_state(package_state: PackageState, current_post_office: PostOffice | None) -> str:
    if current_post_office is not None:
        return current_post_office.name
    locations = {
        PackageState.IN_TRANSIT: "customs_checkpoint",
        PackageState.OUT_FOR_DELIVERY: "delivery_vehicle",
        PackageState.DELIVERED: "recipient_address",
    }
    return locations[package_state]


def _status_history_for_state(package_state: PackageState) -> tuple[str, ...]:
    state_timelines = {
        PackageState.CREATED: ("created",),
        PackageState.SENT: ("created", "sent"),
        PackageState.RECEIVED_AT_ORIGIN_HUB: (
            "created",
            "sent",
            "received_at_origin_hub",
        ),
        PackageState.AT_SORTING_CENTER: (
            "created",
            "sent",
            "received_at_origin_hub",
            "at_sorting_center",
        ),
        PackageState.IN_TRANSIT: (
            "created",
            "sent",
            "received_at_origin_hub",
            "at_sorting_center",
            "in_transit",
        ),
        PackageState.OUT_FOR_DELIVERY: (
            "created",
            "sent",
            "received_at_origin_hub",
            "at_sorting_center",
            "in_transit",
            "out_for_delivery",
        ),
        PackageState.DELIVERED: (
            "created",
            "sent",
            "received_at_origin_hub",
            "at_sorting_center",
            "in_transit",
            "out_for_delivery",
            "delivered",
        ),
        PackageState.RETURNED: (
            "created",
            "sent",
            "received_at_origin_hub",
            "at_sorting_center",
            "returned",
        ),
    }
    return state_timelines[package_state]


def _random_modes() -> tuple[TransportMode, ...]:
    modes = list(TransportMode)
    random.shuffle(modes)
    return tuple(sorted(modes[: random.randint(1, len(modes))], key=lambda mode: mode.label))


def _random_ranges(
    target_range: ShippingRange | None = None,
) -> tuple[ShippingRange, ...]:
    ranges = list(ShippingRange)
    random.shuffle(ranges)
    selected = set(ranges[: random.randint(1, len(ranges))])
    if target_range is not None:
        selected.add(target_range)
    return tuple(sorted(selected, key=lambda shipping_range: shipping_range.label))


def _allowed_modes_for_package(package: Package) -> list[TransportMode]:
    allowed_modes: list[TransportMode] = []
    for mode in TransportMode:
        if package.weight_kg > mode.max_recommended_weight_kg:
            continue
        if package.shipping_range is ShippingRange.INTERCONTINENTAL and mode not in {
            TransportMode.FLIGHT,
            TransportMode.SHIPMENT,
        }:
            continue
        if package.category is PackageCategory.FRAGILE and mode is TransportMode.SHIPMENT:
            continue
        if package.priority is DeliveryPriority.EXPRESS and mode is TransportMode.SHIPMENT:
            continue
        allowed_modes.append(mode)

    return allowed_modes or [TransportMode.TRUCK]


def _preferred_modes_for_shipping_range(
    shipping_range: ShippingRange,
) -> tuple[TransportMode, ...]:
    if shipping_range is ShippingRange.INTERCONTINENTAL:
        return (TransportMode.FLIGHT, TransportMode.SHIPMENT)
    return tuple(TransportMode)


def _supports_shipping_range(candidate_range: ShippingRange, required_range: ShippingRange) -> bool:
    return candidate_range.distance_score >= required_range.distance_score


def _required_shipping_range_for_regions(
    origin_region: Region, destination_region: Region
) -> ShippingRange:
    if origin_region is destination_region:
        return ShippingRange.DOMESTIC
    if origin_region.is_us == destination_region.is_us:
        return ShippingRange.REGIONAL
    return ShippingRange.INTERCONTINENTAL


def _compatible_user_retailer_pairs(
    users: list[User],
    retailers: list[Retailer],
) -> list[tuple[User, Retailer, ShippingRange]]:
    compatible_pairs: list[tuple[User, Retailer, ShippingRange]] = []
    for user in users:
        for retailer in retailers:
            required_range = _required_shipping_range_for_regions(retailer.region, user.region)
            if not _supports_shipping_range(user.default_shipping_range, required_range):
                continue
            if not _supports_shipping_range(retailer.shipping_range, required_range):
                continue
            compatible_pairs.append((user, retailer, required_range))
    return compatible_pairs


def _seed_post_offices(
    shipping_range: ShippingRange, retailer_region: Region, user_region: Region
) -> list[PostOffice]:
    retailer_secondary_region = _secondary_region_for(retailer_region)
    network: list[PostOffice] = [
        _create_post_office(
            name=f"{retailer_region.display_name} Warehouse A",
            office_type=PostOfficeType.ORIGIN_WAREHOUSE,
            region=retailer_region,
            shipping_range=shipping_range,
            supports_cold_chain=True,
            supports_hazardous_materials=False,
        ),
        _create_post_office(
            name=f"{retailer_secondary_region.display_name} Warehouse B",
            office_type=PostOfficeType.ORIGIN_WAREHOUSE,
            region=retailer_secondary_region,
            shipping_range=shipping_range,
            supports_cold_chain=True,
            supports_hazardous_materials=False,
        ),
        _create_post_office(
            name=f"{retailer_region.display_name} Hub",
            office_type=PostOfficeType.ORIGIN_HUB,
            region=retailer_region,
            shipping_range=shipping_range,
            supports_cold_chain=True,
            supports_hazardous_materials=True,
        ),
        _create_post_office(
            name=f"{retailer_secondary_region.display_name} Sorting Center",
            office_type=PostOfficeType.SORTING_CENTER,
            region=retailer_secondary_region,
            shipping_range=shipping_range,
            supports_cold_chain=True,
            supports_hazardous_materials=True,
        ),
    ]

    if user_region is not retailer_secondary_region:
        network.append(
            _create_post_office(
                name=f"{user_region.display_name} Sorting Center",
                office_type=PostOfficeType.SORTING_CENTER,
                region=user_region,
                shipping_range=shipping_range,
                supports_cold_chain=True,
                supports_hazardous_materials=True,
            )
        )

    covered_regions = {post_office.region for post_office in network}
    for region in Region:
        if region in covered_regions:
            continue
        network.append(
            _create_post_office(
                name=f"{region.display_name} Regional Sorting Center",
                office_type=PostOfficeType.SORTING_CENTER,
                region=region,
                shipping_range=shipping_range,
                supports_cold_chain=True,
                supports_hazardous_materials=True,
            )
        )

    return network


def _seed_carrier(
    company_name: str,
    region: Region | None = None,
    shipping_range: ShippingRange | None = None,
    supports_temperature_control: bool | None = None,
    hazardous_certified: bool | None = None,
    supported_modes: tuple[TransportMode, ...] | None = None,
) -> Carrier:
    return Carrier(
        carrier_id=f"carrier-{random.randint(100, 999)}",
        company_name=company_name,
        region=region or random.choice(list(Region)),
        supported_modes=supported_modes or _random_modes(),
        shipping_ranges=_random_ranges(target_range=shipping_range),
        temperature_controlled=supports_temperature_control
        if supports_temperature_control is not None
        else random.choice(transport_possible_values["temperature_controlled"]),
        hazardous_certified=hazardous_certified
        if hazardous_certified is not None
        else random.choice(transport_possible_values["hazardous_certified"]),
        weekend_operations=random.choice(transport_possible_values["weekend_operations"]),
        max_weight_kg=random.choice(transport_possible_values["max_weight_kg"]),
        reliability_score=random.choice(transport_possible_values["reliability_score"]),
        base_delay_days=random.choice(transport_possible_values["base_delay_days"]),
    )


def _seed_world_carriers(
    retailer: Retailer,
    user: User,
    shipping_range: ShippingRange,
) -> list[Carrier]:
    preferred_modes = _preferred_modes_for_shipping_range(shipping_range)
    all_modes = tuple(TransportMode)
    region_mode_overrides = {
        retailer.region: preferred_modes,
        user.region: all_modes,
        Region.GERMANY: (TransportMode.FREIGHT_RAIL,),
    }

    carriers: list[Carrier] = []
    for region in Region:
        supported_modes = region_mode_overrides.get(region, preferred_modes)
        for carrier_index in range(2):
            carriers.append(
                _seed_carrier(
                    company_name=_carrier_name_for_region(region, carrier_index),
                    region=region,
                    shipping_range=shipping_range,
                    supports_temperature_control=True,
                    hazardous_certified=True,
                    supported_modes=supported_modes,
                )
            )

    return carriers


def _seed_retailer(
    shipping_range: str,
    region: str | None = None,
    origin_post_office_id: str | None = None,
    name: str | None = None,
) -> Retailer:
    region = region or random.choice(retailer_possible_values["region"])
    return Retailer.from_labels(
        retailer_id=f"retailer-{random.randint(1000, 9999)}",
        name=name or random.choice(RETAILER_NAMES),
        region=region,
        shipping_range=shipping_range,
        handling_fee=random.choice(retailer_possible_values["handling_fee"]),
        processing_days=random.choice(retailer_possible_values["processing_days"]),
        origin_post_office_id=origin_post_office_id,
    )


def _seed_user(
    weekend_delivery_eligible: bool | None = None,
    user_tier: str | None = None,
    region: str | None = None,
    default_shipping_range: str | None = None,
    name: str | None = None,
) -> User:
    weekend_delivery_eligible = (
        weekend_delivery_eligible
        if weekend_delivery_eligible is not None
        else random.choice(user_possible_values["weekend_delivery_eligible"])
    )
    user_tier = user_tier or random.choice(user_possible_values["user_tier"])
    region = region or random.choice(user_possible_values["region"])
    default_shipping_range = default_shipping_range or random.choice(
        user_possible_values["default_shipping_range"]
    )

    return User.from_labels(
        user_id=f"user-{random.randint(1000, 9999)}",
        name=name or random.choice(USER_NAMES),
        tier=user_tier,
        region=region,
        weekend_delivery_eligible=weekend_delivery_eligible,
        default_shipping_range=default_shipping_range,
    )


def _seed_package(
    post_offices: list[PostOffice] | None = None,
    shipping_range: str | None = None,
    category: str | None = None,
    priority: str | None = None,
    current_state: str | None = None,
    description: str | None = None,
    origin_region: Region | None = None,
    preferred_origin_post_office_id: str | None = None,
    retailer_id: str | None = None,
    retailer_name: str | None = None,
    user_id: str | None = None,
    user_name: str | None = None,
) -> Package:
    shipping_range = shipping_range or random.choice(package_possible_values["shipping_range"])
    category = category or random.choice(package_possible_values["category"])
    priority = priority or random.choice(package_possible_values["priority"])
    if current_state is None:
        current_state = random.choice(package_possible_values["current_state"])
    package_state = PackageState.from_label(current_state)
    visited_post_offices = _route_post_offices_for_state(
        package_state,
        post_offices or [],
        origin_region=origin_region,
        preferred_origin_post_office_id=preferred_origin_post_office_id,
    )
    current_post_office = visited_post_offices[-1] if visited_post_offices else None

    return Package.from_labels(
        package_id=f"pkg-{random.randint(10000, 99999)}",
        description=description or random.choice(PACKAGE_DESCRIPTIONS),
        weight_kg=random.choice(package_possible_values["weight_kg"]),
        shipping_range=shipping_range,
        category=category,
        priority=priority,
        retailer_id=retailer_id,
        retailer_name=retailer_name,
        user_id=user_id,
        user_name=user_name,
        insured=random.choice(package_possible_values["insured"]),
        current_state=current_state,
        last_known_location=_location_for_state(package_state, current_post_office),
        current_post_office_id=current_post_office.post_office_id if current_post_office else None,
        current_post_office_name=current_post_office.name if current_post_office else None,
        route_post_office_ids=tuple(
            post_office.post_office_id for post_office in visited_post_offices
        ),
        route_post_office_names=tuple(post_office.name for post_office in visited_post_offices),
        status_history=_status_history_for_state(package_state),
    )


def pretty_print_world(world: dict[str, object]) -> str:
    retailers = list(world.get("retailers", []))
    users = list(world.get("users", []))
    packages = list(world.get("packages", []))

    post_offices = world["post_offices"]
    carriers = world["carriers"]

    lines = [
        "World Snapshot",
        "Retailers In World",
    ]

    for index, retailer in enumerate(retailers, start=1):
        lines.extend(
            [
                f"  - Retailer {index}: {retailer.name}",
                f"    ID: {retailer.retailer_id}",
                f"    Region: {retailer.region.display_name}",
                f"    Shipping Range: {retailer.shipping_range.label}",
                f"    Handling Fee: ${retailer.handling_fee:.2f}",
                f"    Processing Days: {retailer.processing_days}",
                f"    Origin Post Office ID: {retailer.origin_post_office_id}",
            ]
        )

    lines.extend(
        [
            "Users In World",
        ]
    )

    for index, user in enumerate(users, start=1):
        lines.extend(
            [
                f"  - User {index}: {user.name}",
                f"    ID: {user.user_id}",
                f"    Tier: {user.tier.label}",
                f"    Region: {user.region.display_name}",
                f"    Weekend Delivery Eligible: {user.weekend_delivery_eligible}",
                f"    Default Shipping Range: {user.default_shipping_range.label}",
            ]
        )

    lines.extend(
        [
            "Packages In World",
        ]
    )

    for index, package in enumerate(packages, start=1):
        lines.extend(
            [
                f"  - Package {index}: {package.package_id}",
                f"    Description: {package.description}",
                f"    Retailer: {package.retailer_name} ({package.retailer_id})"
                if package.retailer_name or package.retailer_id
                else "    Retailer: none",
                f"    User: {package.user_name} ({package.user_id})"
                if package.user_name or package.user_id
                else "    User: none",
                f"    Weight (kg): {package.weight_kg}",
                f"    Shipping Range: {package.shipping_range.label}",
                f"    Category: {package.category.label}",
                f"    Priority: {package.priority.label}",
                f"    Insured: {package.insured}",
                f"    Current State: {package.current_state.label}",
                f"    Last Known Location: {package.last_known_location}",
                f"    Current Post Office: {package.current_post_office_name}",
                f"    Post Office Route: {', '.join(package.route_post_office_names) if package.route_post_office_names else 'none'}",
                f"    Status History: {', '.join(package.status_history) if package.status_history else 'none'}",
            ]
        )

    lines.extend(
        [
            "Post Offices",
        ]
    )

    for post_office in post_offices:
        lines.extend(
            [
                f"  - {post_office.name}",
                f"    ID: {post_office.post_office_id}",
                f"    Type: {post_office.office_type.label}",
                f"    Region: {post_office.region.display_name}",
                f"    Shipping Range: {post_office.shipping_range.label}",
                f"    Cold Chain: {post_office.supports_cold_chain}",
                f"    Hazardous Materials: {post_office.supports_hazardous_materials}",
            ]
        )

    lines.append("Carriers")
    for carrier in carriers:
        lines.extend(
            [
                f"  - {carrier.company_name}",
                f"    ID: {carrier.carrier_id}",
                f"    Region: {carrier.region.display_name}",
                f"    Modes: {', '.join(mode.label for mode in carrier.supported_modes)}",
                f"    Shipping Ranges: {', '.join(shipping_range.label for shipping_range in carrier.shipping_ranges)}",
                f"    Temperature Controlled: {carrier.temperature_controlled}",
                f"    Hazardous Certified: {carrier.hazardous_certified}",
                f"    Weekend Operations: {carrier.weekend_operations}",
                f"    Max Weight (kg): {carrier.max_weight_kg}",
                f"    Reliability Score: {carrier.reliability_score}",
                f"    Base Delay Days: {carrier.base_delay_days}",
            ]
        )

    return "\n".join(lines)
