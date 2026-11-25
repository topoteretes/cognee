from cognee.modules.users.exceptions import CapabilityNotFoundError
from cognee.modules.users.permissions.permission_types import CAPABILITY_TYPES


def validate_capability(capabilities: list[str] | str) -> None:
    """
        Reject anything outside the catalog, including dataset permission names.

        read/write/delete/share are scoped to a dataset through ACL and do not
        belong here; anything else is a name nothing in the code gates, and
        storing it would look like it worked while granting nothing.

        Every name is checked before any is accepted, and all unknown ones are
        reported together, so a batch is rejected whole rather than half
        applied.
    Args:
        capabilities: Names to check against the CAPABILITY_TYPES catalog.

    Raises:
        CapabilityNotFoundError: If any name is not in the catalog.
    """
    # If only a single capability is provided transform it to a list
    if not isinstance(capabilities, list):
        capabilities = [capabilities]

    unknown = [capability for capability in capabilities if capability not in CAPABILITY_TYPES]
    if unknown:
        raise CapabilityNotFoundError(
            message=f"Unknown capability: {', '.join(unknown)}. "
            f"Known: {', '.join(sorted(CAPABILITY_TYPES))}"
        )
