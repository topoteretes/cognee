import random
import string
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from cognee.infrastructure.engine import DataPoint


def run_test_against_ground_truth(
    test_target_item_name: str, test_target_item: Any, ground_truth_dict: Dict[str, Any]
):
    """Validates test target item attributes against ground truth values.

    Args:
        test_target_item_name: Name of the item being tested (for error messages)
        test_target_item: Object whose attributes are being validated
        ground_truth_dict: Dictionary containing expected values

    Raises:
        AssertionError: If any attribute doesn't match ground truth or if update timestamp is too old
    """
    for key, ground_truth in ground_truth_dict.items():
        if isinstance(ground_truth, dict):
            for key2, ground_truth2 in ground_truth.items():
                assert (
                    ground_truth2 == getattr(test_target_item, key)[key2]
                ), f"{test_target_item_name}/{key = }/{key2 = }: {ground_truth2 = } != {getattr(test_target_item, key)[key2] = }"
        elif isinstance(ground_truth, list):
            raise NotImplementedError("Currently not implemented for 'list'")
        else:
            assert ground_truth == getattr(
                test_target_item, key
            ), f"{test_target_item_name}/{key = }: {ground_truth = } != {getattr(test_target_item, key) = }"
    time_delta = datetime.now(timezone.utc) - getattr(test_target_item, "updated_at")

    assert time_delta.total_seconds() < 60, f"{ time_delta.total_seconds() = }"


class Organization(DataPoint):
    id: str
    name: str
    members: Optional[list["SocietyPerson"]]


class SocietyPerson(DataPoint):
    id: str
    name: str
    memberships: Optional[list[Organization]]


SocietyPerson.model_rebuild()
Organization.model_rebuild()


ORGANIZATION_NAMES = [
    "ChessClub",
    "RowingClub",
    "TheatreTroupe",
    "PoliticalParty",
    "Charity",
    "FanClub",
    "FilmClub",
    "NeighborhoodGroup",
    "LocalCouncil",
    "Band",
]
PERSON_NAMES = ["Sarah", "Anna", "John", "Sam"]


def create_society_person_recursive(id, name, organization_names, max_depth, depth=0):
    id_suffix = "".join(random.choice(string.ascii_lowercase) for _ in range(10))

    if depth < max_depth:
        memberships = [
            create_organization_recursive(
                f"{org_name}-{depth}-{id_suffix}",
                org_name.lower(),
                PERSON_NAMES,
                max_depth,
                depth + 1,
            )
            for org_name in organization_names
        ]
    else:
        memberships = None

    return SocietyPerson(id=id, name=f"{name}{depth}", memberships=memberships)


def create_organization_recursive(id, name, member_names, max_depth, depth=0):
    id_suffix = "".join(random.choice(string.ascii_lowercase) for _ in range(10))

    if depth < max_depth:
        members = [
            create_society_person_recursive(
                f"{member_name}-{depth}-{id_suffix}",
                member_name.lower(),
                ORGANIZATION_NAMES,
                max_depth,
                depth + 1,
            )
            for member_name in member_names
        ]
    else:
        members = None

    return Organization(id=id, name=f"{name}{depth}", members=members)


def count_society(obj):
    if isinstance(obj, SocietyPerson):
        if obj.memberships is not None:
            organization_counts, society_person_counts = zip(
                *[count_society(organization) for organization in obj.memberships]
            )
            organization_count = sum(organization_counts)
            society_person_count = sum(society_person_counts) + 1
            return (organization_count, society_person_count)
        else:
            return (0, 1)
    if isinstance(obj, Organization):
        if obj.members is not None:
            organization_counts, society_person_counts = zip(
                *[count_society(organization) for organization in obj.members]
            )
            organization_count = sum(organization_counts) + 1
            society_person_count = sum(society_person_counts)
            return (organization_count, society_person_count)
        else:
            return (1, 0)
    else:
        raise Exception("Not allowed")


def show_first_difference(str1, str2, str1_name, str2_name, context=30):
    for i, (c1, c2) in enumerate(zip(str1, str2)):
        if c1 != c2:
            start = max(0, i - context)
            end1 = min(len(str1), i + context + 1)
            end2 = min(len(str2), i + context + 1)
            if i > 0:
                return f"identical: '{str1[start:i-1]}' | {str1_name}: '{str1[i-1:end1]}'... != {str2_name}: '{str2[i-1:end2]}'..."
            else:
                return f"{str1_name} and {str2_name} have no overlap in characters"
    if len(str1) > len(str2):
        return f"{str2_name} is identical up to the {i}th character, missing afterwards '{str1[i:i+context]}'..."
    if len(str2) > len(str1):
        return f"{str1_name} is identical up to the {i}th character, missing afterwards '{str2[i:i+context]}'..."
    else:
        return f"{str1_name} and {str2_name} are identical."
