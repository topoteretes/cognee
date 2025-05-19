import duckdb
import pathlib
from os import path


def extract_dataset_instance(
    dataset_file_path: str, destination_directory: str, number_of_instances: int = 10
) -> None:
    connection = duckdb.connect()
    df = connection.execute(f"SELECT * FROM '{dataset_file_path}'").fetchdf()
    extracted_instances = 0

    for _, row in df.iterrows():
        if extracted_instances == number_of_instances:
            break

        file_id = str(row["id"])
        content = row["text"]

        if not path.exists(destination_directory):
            pathlib.Path(destination_directory).mkdir(parents=True, exist_ok=True)

        with open(path.join(destination_directory, f"{file_id}.txt"), "w") as file:
            file.write(content)

        extracted_instances += 1


if __name__ == "__main__":
    data_directory_path = path.join(pathlib.Path(__file__).parent.parent, ".data")
    dataset_file_name = "de-00000-of-00003-f8e581c008ccc7f2.parquet"
    dataset_file_path = path.join(data_directory_path, dataset_file_name)
    destination_directory_path = path.join(data_directory_path, "instances")
    extract_dataset_instance(dataset_file_path, destination_directory_path, 1)
