from typing import List
import os

from cognee.infrastructure.files.storage import get_storage_config
from cognee.modules.users.models import User
from cognee.modules.data.models import Data
import dlt

import tempfile


async def ingest_dlt_source(
    dlt_source,
    dataset_name: str,
    user: User = None,
) -> List[Data]:
    storage_config = get_storage_config()
    os.environ["BUCKET_URL"] = storage_config["data_root_directory"]

    dlt_source.max_table_nesting = 0
    pipeline = dlt.pipeline(
        pipeline_name="ingest_dlt_source",
        destination="filesystem",
        dataset_name=dataset_name,
    )

    load_info = pipeline.run(dlt_source, write_disposition="replace")

    result_file_names = []
    for table in load_info.load_packages[0].schema.tables.keys():
        if "_dlt_" not in table:
            file_id = ""
            for job in load_info.load_packages[0].jobs["completed_jobs"]:
                if job.job_file_info.table_name == table:
                    file_id = job.job_file_info.file_id
                    break
            filename = (
                storage_config["data_root_directory"]
                + "/"
                + dlt_source.name
                + "/"
                + table
                + "/"
                + load_info.load_packages[0].load_id
                + "."
                + file_id
                + ".jsonl"
            )
            result_file_names.append(filename)

    # keys_to_drop = ["_dlt_load_id", "_dlt_id"]

    # for result_file in result_file_names:
    #     remove_columns_jsonl_inplace(result_file)

    return result_file_names


# Use orjson if available for speed; fall back to built-in json
try:
    import orjson

    def _loads(b):
        return orjson.loads(b)

    def _dumps_line(obj):
        return orjson.dumps(obj) + b"\n"
except Exception:
    import json

    def _loads(b):
        return json.loads(b.decode("utf-8"))

    def _dumps_line(obj):
        s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        return (s + "\n").encode("utf-8")


def remove_columns_jsonl_inplace(path, drop_keys="", skip_invalid=False):
    """
    Remove given top-level keys from each JSON object line in a .jsonl file.
    Edits the file in place via a temp file and atomic replace.
    """
    drop_keys = set(drop_keys)
    dir_ = os.path.dirname(path) or "."

    with open(path, "rb") as src, tempfile.NamedTemporaryFile("wb", dir=dir_, delete=False) as dst:
        tmp = dst.name
        for line in src:
            if not line.strip():
                continue  # skip blank lines
            try:
                obj = _loads(line)
            except Exception:
                if skip_invalid:
                    continue
                raise
            if isinstance(obj, dict):
                for key in obj.keys():
                    if "_dlt_" in key:
                        drop_keys.add(key)
                for k in drop_keys:
                    obj.pop(k, None)
            dst.write(_dumps_line(obj))
    os.replace(tmp, path)  # atomic on POSIX and Windows
