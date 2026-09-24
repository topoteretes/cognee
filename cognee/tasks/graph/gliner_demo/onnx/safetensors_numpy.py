"""Read a ``.safetensors`` file into numpy arrays without torch or the safetensors package.

The format (https://github.com/huggingface/safetensors) is an 8-byte little-endian
header length, a JSON header mapping each tensor name to its dtype, shape and
byte range, then the raw little-endian tensor bytes. Arrays are memory-mapped
views of the file, so reading a 738 MB checkpoint copies nothing.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

_DTYPES = {
    "F64": np.float64,
    "F32": np.float32,
    "F16": np.float16,
    "I64": np.int64,
    "I32": np.int32,
    "I16": np.int16,
    "I8": np.int8,
    "U8": np.uint8,
    "BOOL": np.bool_,
}


def load(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    with path.open("rb") as handle:
        (header_size,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_size))
    data = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + header_size)
    tensors = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        dtype = _DTYPES.get(entry["dtype"])
        if dtype is None:
            raise ValueError(f"{path.name}: tensor {name} has unsupported dtype {entry['dtype']}")
        start, end = entry["data_offsets"]
        tensors[name] = (
            data[start:end].view(np.dtype(dtype).newbyteorder("<")).reshape(entry["shape"])
        )
    return tensors
