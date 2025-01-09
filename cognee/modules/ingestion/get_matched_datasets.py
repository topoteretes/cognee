from .discover_directory_datasets import discover_directory_datasets


def get_matched_datasets(data_path: str, dataset_name_to_match: str = None):
    datasets = discover_directory_datasets(data_path)

    matched_datasets = []

    for dataset_name, dataset_files in datasets.items():
        if dataset_name_to_match is None or dataset_name.startswith(dataset_name_to_match):
            matched_datasets.append([dataset_name, dataset_files])

    return matched_datasets
