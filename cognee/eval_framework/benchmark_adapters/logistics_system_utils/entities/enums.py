from enum import Enum


class UserTier(Enum):
    STANDARD = ("standard", 0, 1)
    BUSINESS = ("business", 1, 2)
    ENTERPRISE = ("enterprise", 2, 3)

    @property
    def label(self) -> str:
        return self.value[0]

    @property
    def priority_bonus(self) -> int:
        return self.value[1]

    @property
    def speed_weight(self) -> int:
        return self.value[2]

    @classmethod
    def from_label(cls, label: str) -> "UserTier":
        normalized = label.lower()
        for item in cls:
            if item.label == normalized:
                return item
        raise ValueError(f"Unknown label: {label}")


class ShippingRange(Enum):
    DOMESTIC = ("domestic", 0)
    REGIONAL = ("regional", 2)
    INTERCONTINENTAL = ("intercontinental", 6)

    @property
    def label(self) -> str:
        return self.value[0]

    @property
    def distance_score(self) -> int:
        return self.value[1]

    @classmethod
    def from_label(cls, label: str) -> "ShippingRange":
        normalized = label.lower()
        for item in cls:
            if item.label == normalized:
                return item
        raise ValueError(f"Unknown label: {label}")


class Region(Enum):
    US_NORTH = ("us_north", "US North", 4, True)
    US_SOUTH = ("us_south", "US South", 0, True)
    US_EAST = ("us_east", "US East", 3, True)
    US_WEST = ("us_west", "US West", 1, True)
    US_CENTRAL = ("us_central", "US Central", 2, True)
    GERMANY = ("germany", "Germany", 2, False)
    FRANCE = ("france", "France", 2, False)
    NETHERLANDS = ("netherlands", "Netherlands", 1, False)
    SPAIN = ("spain", "Spain", 0, False)
    ITALY = ("italy", "Italy", 1, False)
    POLAND = ("poland", "Poland", 3, False)

    @property
    def label(self) -> str:
        return self.value[0]

    @property
    def display_name(self) -> str:
        return self.value[1]

    @property
    def distance_score(self) -> int:
        return self.value[2]

    @property
    def is_us(self) -> bool:
        return self.value[3]

    @classmethod
    def from_label(cls, label: str) -> "Region":
        normalized = label.lower()
        for item in cls:
            if item.label == normalized:
                return item
        raise ValueError(f"Unknown label: {label}")


class DeliveryPriority(Enum):
    ECONOMY = ("economy", 1, 3)
    STANDARD = ("standard", 2, 2)
    EXPRESS = ("express", 3, 1)

    @property
    def label(self) -> str:
        return self.value[0]

    @property
    def speed_weight(self) -> int:
        return self.value[1]

    @property
    def cost_weight(self) -> int:
        return self.value[2]

    @classmethod
    def from_label(cls, label: str) -> "DeliveryPriority":
        normalized = label.lower()
        for item in cls:
            if item.label == normalized:
                return item
        raise ValueError(f"Unknown label: {label}")


class TransportMode(Enum):
    FLIGHT = ("flight", 1, 5, 500.0)
    SHIPMENT = ("shipment", 7, 1, 20000.0)
    FREIGHT_RAIL = ("freight_rail", 4, 2, 12000.0)
    TRUCK = ("truck", 3, 3, 8000.0)

    @property
    def label(self) -> str:
        return self.value[0]

    @property
    def transit_days(self) -> int:
        return self.value[1]

    @property
    def cost_score(self) -> int:
        return self.value[2]

    @property
    def max_recommended_weight_kg(self) -> float:
        return self.value[3]

    @classmethod
    def from_label(cls, label: str) -> "TransportMode":
        normalized = label.lower()
        for item in cls:
            if item.label == normalized:
                return item
        raise ValueError(f"Unknown label: {label}")


class PackageCategory(Enum):
    STANDARD = ("standard", False, False, False)
    FRAGILE = ("fragile", False, False, True)
    PERISHABLE = ("perishable", True, False, False)
    HAZARDOUS = ("hazardous", False, True, False)

    @property
    def label(self) -> str:
        return self.value[0]

    @property
    def requires_temperature_control(self) -> bool:
        return self.value[1]

    @property
    def requires_hazard_certification(self) -> bool:
        return self.value[2]

    @property
    def is_fragile(self) -> bool:
        return self.value[3]

    @classmethod
    def from_label(cls, label: str) -> "PackageCategory":
        normalized = label.lower()
        for item in cls:
            if item.label == normalized:
                return item
        raise ValueError(f"Unknown label: {label}")


class PackageState(Enum):
    CREATED = ("created", False, False, False)
    SENT = ("sent", True, False, False)
    RECEIVED_AT_ORIGIN_HUB = ("received_at_origin_hub", True, False, False)
    AT_SORTING_CENTER = ("at_sorting_center", True, False, True)
    IN_TRANSIT = ("in_transit", True, False, False)
    OUT_FOR_DELIVERY = ("out_for_delivery", True, False, False)
    DELIVERED = ("delivered", True, True, False)
    RETURNED = ("returned", True, False, False)

    @property
    def label(self) -> str:
        return self.value[0]

    @property
    def has_been_sent(self) -> bool:
        return self.value[1]

    @property
    def is_received_by_customer(self) -> bool:
        return self.value[2]

    @property
    def is_at_sorting_center(self) -> bool:
        return self.value[3]

    @classmethod
    def from_label(cls, label: str) -> "PackageState":
        normalized = label.lower()
        for item in cls:
            if item.label == normalized:
                return item
        raise ValueError(f"Unknown label: {label}")


class PostOfficeType(Enum):
    ORIGIN_WAREHOUSE = "origin_warehouse"
    ORIGIN_HUB = "origin_hub"
    SORTING_CENTER = "sorting_center"

    @property
    def label(self) -> str:
        return self.value

    @classmethod
    def from_label(cls, label: str) -> "PostOfficeType":
        normalized = label.lower()
        for item in cls:
            if item.label == normalized:
                return item
        raise ValueError(f"Unknown label: {label}")


def us_regions() -> tuple[Region, ...]:
    return tuple(region for region in Region if region.is_us)


# Backward-compatible aliases for the earlier prototype vocabulary.
CustomerTier = UserTier
TransportationMode = TransportMode
