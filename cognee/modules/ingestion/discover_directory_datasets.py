from os import path, listdir


def discover_directory_datasets(root_dir_path: str, parent_dir: str = None):
    datasets = {}

    for file_or_dir in listdir(root_dir_path):
        if path.isdir(path.join(root_dir_path, file_or_dir)):
            dataset_name = file_or_dir if parent_dir is None else f"{parent_dir}.{file_or_dir}"

            nested_datasets = discover_directory_datasets(
                path.join(root_dir_path, file_or_dir), dataset_name
            )

            for dataset in nested_datasets.keys():
                datasets[dataset] = nested_datasets[dataset]
        else:
            if parent_dir not in datasets:
                datasets[parent_dir] = []

            datasets[parent_dir].append(path.join(root_dir_path, file_or_dir))

    return datasets
