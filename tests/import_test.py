def test_import_cognee():
    try:
        import cognee

        assert True  # Pass the test if no error occurs
    except ImportError as e:
        assert False, f"Failed to import cognee: {e}"
