from cognee.root_dir import get_absolute_path
import json
import requests
from jsonschema import ValidationError, validate
from pathlib import Path


qa_datasets = {
    "hotpotqa": {
        "filename": "hotpot_dev_fullwiki_v1.json",
        "URL": "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json",
    },
    "2wikimultihop": {
        "filename": "data/dev.json",
        "URL": "https://www.dropbox.com/scl/fi/heid2pkiswhfaqr5g0piw/data.zip?rlkey=ira57daau8lxfj022xvk1irju&e=1",
    },
}

qa_json_schema = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "question": {"type": "string"},
            "context": {"type": "array"},
        },
        "required": ["answer", "question", "context"],
        "additionalProperties": True,
    },
}


def download_qa_dataset(dataset_name: str, filepath: Path):
    if dataset_name not in qa_datasets:
        raise ValueError(f"{dataset_name} is not a supported dataset.")

    url = qa_datasets[dataset_name]["URL"]

    if dataset_name == "2wikimultihop":
        raise Exception(
            "Please download 2wikimultihop dataset (data.zip) manually from \
                        https://www.dropbox.com/scl/fi/heid2pkiswhfaqr5g0piw/data.zip?rlkey=ira57daau8lxfj022xvk1irju&e=1 \
                        and unzip it."
        )

    response = requests.get(url, stream=True)

    if response.status_code == 200:
        with open(filepath, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Dataset {dataset_name} downloaded and saved to {filepath}")
    else:
        print(f"Failed to download {dataset_name}. Status code: {response.status_code}")


def load_qa_dataset(dataset_name_or_filename: str) -> list[dict]:
    if dataset_name_or_filename in qa_datasets:
        dataset_name = dataset_name_or_filename
        filename = qa_datasets[dataset_name]["filename"]

        data_root_dir = get_absolute_path("../.data")
        if not Path(data_root_dir).exists():
            Path(data_root_dir).mkdir()

        filepath = data_root_dir / Path(filename)
        if not filepath.exists():
            download_qa_dataset(dataset_name, filepath)
    else:
        filename = dataset_name_or_filename
        filepath = Path(filename)

    with open(filepath, "r") as file:
        dataset = json.load(file)

    try:
        validate(instance=dataset, schema=qa_json_schema)
    except ValidationError as e:
        raise ValidationError(f"Invalid QA dataset: {e.message}")

    return dataset
