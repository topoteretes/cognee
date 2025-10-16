# C# and C++ Dependency Extraction Tests

This directory contains comprehensive unit tests for verifying the codegraph pipeline's ability to correctly parse, detect, and extract class and function information from C# and C++ source files.

## Test Structure

```
repo_processor/
├── __init__.py                         # Package initialization
├── test_csharp_cpp_dependencies.py     # Main test suite
├── fixtures/                           # Test fixture files
│   ├── __init__.py
│   ├── sample_csharp.cs               # C# sample code
│   └── sample_cpp.cpp                 # C++ sample code
└── README.md                          # This file
```

## Test Coverage

### C# Tests (`TestCSharpDependencies`)

The test suite for C# dependency extraction verifies the following capabilities:

1. **Class Extraction** (`test_csharp_class_extraction`)
   - Detects public classes
   - Identifies class names correctly
   - Handles multiple classes in a single file

2. **Function/Method Extraction** (`test_csharp_function_extraction`)
   - Extracts public and private methods
   - Captures method signatures
   - Handles async methods
   - Detects constructors

3. **Interface Extraction** (`test_csharp_interface_extraction`)
   - Identifies interface definitions
   - Extracts interface members
   - Handles generic interfaces

4. **Using Statements** (`test_csharp_using_statements`)
   - Detects import statements (using directives)
   - Captures namespace dependencies
   - Identifies external library usage

### C++ Tests (`TestCppDependencies`)

The test suite for C++ dependency extraction verifies the following capabilities:

1. **Class Extraction** (`test_cpp_class_extraction`)
   - Detects class definitions
   - Handles nested classes
   - Identifies template classes

2. **Function Extraction** (`test_cpp_function_extraction`)
   - Extracts member functions (methods)
   - Captures global functions
   - Handles function overloading
   - Detects template functions

3. **Namespace Extraction** (`test_cpp_namespace_extraction`)
   - Identifies namespace declarations
   - Handles nested namespaces
   - Captures namespace-scoped entities

4. **Include Statements** (`test_cpp_include_statements`)
   - Detects #include directives
   - Distinguishes between system and local headers
   - Identifies external dependencies

5. **Inheritance Detection** (`test_cpp_inheritance`)
   - Detects base classes
   - Identifies inheritance relationships
   - Handles multiple inheritance

### Integration Tests (`TestIntegration`)

These tests verify end-to-end functionality:

1. **C# File Processing** (`test_csharp_file_processing`)
   - Tests complete pipeline for .cs files
   - Verifies all extraction components work together

2. **C++ File Processing** (`test_cpp_file_processing`)
   - Tests complete pipeline for .cpp files
   - Verifies all extraction components work together

## Sample Files

The `fixtures/` directory contains comprehensive sample files:

### sample_csharp.cs

Includes examples of:
- Multiple namespaces and classes
- Inheritance hierarchies
- Interfaces and implementations
- Generic types
- Async methods
- Static utility classes
- XML documentation comments
- Using directives

### sample_cpp.cpp

Includes examples of:
- Multiple namespaces
- Class hierarchies with inheritance
- Template classes and functions
- Member functions (methods)
- Global functions
- Include directives
- Comments and documentation
- Multiple inheritance
- Operator overloading

## Running the Tests

### Run all tests in this module:
```bash
pytest cognee/tests/tasks/repo_processor/
```

### Run specific test classes:
```bash
# C# tests only
pytest cognee/tests/tasks/repo_processor/test_csharp_cpp_dependencies.py::TestCSharpDependencies

# C++ tests only
pytest cognee/tests/tasks/repo_processor/test_csharp_cpp_dependencies.py::TestCppDependencies

# Integration tests only
pytest cognee/tests/tasks/repo_processor/test_csharp_cpp_dependencies.py::TestIntegration
```

### Run specific tests:
```bash
pytest cognee/tests/tasks/repo_processor/test_csharp_cpp_dependencies.py::TestCSharpDependencies::test_csharp_class_extraction
```

### Run with verbose output:
```bash
pytest -v cognee/tests/tasks/repo_processor/
```

### Run with coverage:
```bash
pytest --cov=cognee.tasks.repo_processor cognee/tests/tasks/repo_processor/
```

## Test Implementation Details

### Fixtures

Each test class uses pytest fixtures to create temporary files:
- `sample_csharp_file`: Creates a temporary .cs file for testing
- `sample_cpp_file`: Creates a temporary .cpp file for testing

These fixtures automatically clean up after tests complete.

### Assertions

Tests verify that:
1. Dependency extraction functions return valid results (not None)
2. Expected classes, functions, and other code elements are detected
3. Import/include statements are captured
4. The structure of extracted data matches expectations

## Expected Output Format

The dependency extraction functions should return structured data containing:
- **classes**: List of detected classes with their properties
- **functions**: List of detected functions/methods
- **imports**: List of import/include statements
- **namespaces**: Namespace information (for C++)
- **interfaces**: Interface definitions (for C#)

## Dependencies

These tests require:
- pytest
- tree-sitter (for C# and C++ parsing)
- tree-sitter-c-sharp
- tree-sitter-cpp

## Troubleshooting

If tests fail:
1. Ensure tree-sitter parsers are properly installed
2. Verify that the dependency extraction functions exist in the codebase
3. Check that the test fixtures are being created properly
4. Run tests with `-v` flag for more detailed output
5. Check that file paths are correct for your system

## Future Enhancements

Potential additions to the test suite:
- Property extraction for C#
- Attribute/annotation detection
- Documentation comment parsing
- More complex inheritance scenarios
- Generic type parameter extraction
- Lambda and anonymous function detection
- Preprocessor directive handling for C++
