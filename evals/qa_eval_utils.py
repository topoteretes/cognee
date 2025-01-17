import itertools
import matplotlib.pyplot as plt
from jsonschema import ValidationError, validate
import pandas as pd
from pathlib import Path

paramset_json_schema = {
    "type": "object",
    "properties": {
        "dataset": {
            "type": "array",
            "items": {"type": "string"},
        },
        "rag_option": {
            "type": "array",
            "items": {"type": "string"},
        },
        "num_samples": {
            "type": "array",
            "items": {"type": "integer", "minimum": 1},
        },
        "metric_names": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["dataset", "rag_option", "num_samples", "metric_names"],
    "additionalProperties": False,
}


def save_table_as_image(df, image_path):
    plt.figure(figsize=(10, 6))
    plt.axis("tight")
    plt.axis("off")
    plt.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc="center")
    plt.title(f"{df.index.name}")
    plt.savefig(image_path, bbox_inches="tight")
    plt.close()


def save_results_as_image(results, out_path):
    for dataset, num_samples_data in results.items():
        for num_samples, table_data in num_samples_data.items():
            df = pd.DataFrame.from_dict(table_data, orient="index")
            df.index.name = f"Dataset: {dataset}, Num Samples: {num_samples}"
            image_path = out_path / Path(f"table_{dataset}_{num_samples}.png")
            save_table_as_image(df, image_path)


def get_combinations(parameters):
    try:
        validate(instance=parameters, schema=paramset_json_schema)
    except ValidationError as e:
        raise ValidationError(f"Invalid parameter set: {e.message}")

    params_for_combos = {k: v for k, v in parameters.items() if k != "metric_name"}
    keys, values = zip(*params_for_combos.items())
    combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    return combinations
