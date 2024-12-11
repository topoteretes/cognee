import os
from typing import List, Union, BinaryIO

def resolve_data_directories(data: Union[BinaryIO, List[BinaryIO], str, List[str]], include_subdirectories: bool = True):
    """
    Resolves directories by replacing them with their contained files.

    Args:
        data: A single file, directory, or binary stream, or a list of such items.
        include_subdirectories: Whether to include files in subdirectories recursively.

    Returns:
        A list of resolved files and binary streams.
    """
    # Ensure `data` is a list
    if not isinstance(data, list):
        data = [data]

    resolved_data = []

    for item in data:
        if isinstance(item, str):  # Check if the item is a path
            if os.path.isdir(item):  # If it's a directory
                if include_subdirectories:
                    # Recursively add all files in the directory and subdirectories
                    for root, _, files in os.walk(item):
                        resolved_data.extend([os.path.join(root, f) for f in files])
                else:
                    # Add all files (not subdirectories) in the directory
                    resolved_data.extend(
                        [os.path.join(item, f) for f in os.listdir(item) if os.path.isfile(os.path.join(item, f))]
                    )
            elif os.path.isfile(item):  # If it's a file, add it to the resolved_data list
                resolved_data.append(item)
            else:
                raise ValueError(f"Path '{item}' is neither a file nor a directory.")
        elif isinstance(item, BinaryIO):  # If it's a binary stream, add it directly
            resolved_data.append(item)
        else:
            raise TypeError(f"Unsupported type: {type(item)}. Expected str or BinaryIO.")

    return resolved_data
