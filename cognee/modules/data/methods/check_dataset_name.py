import re

from cognee.modules.data.exceptions import InvalidDatasetNameError


def check_dataset_name(dataset_name: str):
    if "." in dataset_name or " " in dataset_name:
        suggestion = re.sub(r"[ .]+", "_", dataset_name.strip(" ."))
        raise InvalidDatasetNameError(
            f"Dataset name cannot contain spaces or dots: '{dataset_name}'.",
            remediation=f"Use '{suggestion}' instead." if suggestion else None,
        )
