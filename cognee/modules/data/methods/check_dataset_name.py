def check_dataset_name(dataset_name: str):
    if "." in dataset_name or " " in dataset_name:
        raise ValueError(f"Dataset name cannot contain spaces or underscores, got {dataset_name}")
