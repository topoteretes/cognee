"""
Test runner and utilities for CLI tests.
"""

import pytest
import sys
from pathlib import Path


def run_cli_tests():
    """Run all CLI tests"""
    test_dir = Path(__file__).parent
    integration_dir = test_dir.parent.parent / "integration" / "cli"

    cli_unit_test_files = [
        "test_cli_main.py",
        "test_cli_commands.py",
        "test_cli_utils.py",
        "test_cli_edge_cases.py",
    ]

    cli_integration_test_files = ["test_cli_integration.py"]

    # Run tests with pytest
    args = ["-v", "--tb=short"]

    # Add unit tests
    for test_file in cli_unit_test_files:
        test_path = test_dir / test_file
        if test_path.exists():
            args.append(str(test_path))

    # Add integration tests
    for test_file in cli_integration_test_files:
        test_path = integration_dir / test_file
        if test_path.exists():
            args.append(str(test_path))

    return pytest.main(args)


def run_specific_cli_test(test_file):
    """Run a specific CLI test file"""
    test_dir = Path(__file__).parent
    test_path = test_dir / test_file

    if not test_path.exists():
        print(f"Test file {test_file} not found")
        return 1

    return pytest.main(["-v", "--tb=short", str(test_path)])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Run specific test file
        exit_code = run_specific_cli_test(sys.argv[1])
    else:
        # Run all CLI tests
        exit_code = run_cli_tests()

    sys.exit(exit_code)
