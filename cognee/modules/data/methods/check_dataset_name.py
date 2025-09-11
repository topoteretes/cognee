def check_dataset_name(dataset_name: str):
    if "." in dataset_name or " " in dataset_name:
        raise ValueError("Dataset name cannot contain spaces or dots.")
