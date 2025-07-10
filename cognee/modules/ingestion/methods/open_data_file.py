def open_data_file(file_path: str, s3fs):
    if file_path.startswith("s3://"):
        return s3fs.open(file_path, mode="rb")
    else:
        local_path = file_path.replace("file://", "")
        return open(local_path, mode="rb")
