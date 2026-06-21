from cognee.infrastructure.files.utils.get_data_file_path import get_data_file_path


def test_get_data_file_path_decodes_s3_key_paths():
    assert (
        get_data_file_path("s3://bucket/folder/my%20file%2Bv1.txt")
        == "s3://bucket/folder/my file+v1.txt"
    )


def test_get_data_file_path_normalizes_decoded_s3_paths():
    assert (
        get_data_file_path("s3://bucket/folder%20name/../other%20folder/file.txt")
        == "s3://bucket/other folder/file.txt"
    )
