import pytest
import tempfile
import os
from pathlib import Path
from cognee.tasks.repo_processor.get_csharp_dependencies import get_csharp_dependencies
from cognee.tasks.repo_processor.get_cpp_dependencies import get_cpp_dependencies


class TestCSharpDependencies:
    """Test suite for C# dependency extraction."""

    @pytest.fixture
    def sample_csharp_file(self):
        """Create a temporary C# file with sample code."""
        code = '''
using System;
using System.Collections.Generic;
using MyNamespace.Models;

namespace TestNamespace
{
    public class Calculator
    {
        private int result;

        public Calculator()
        {
            result = 0;
        }

        public int Add(int a, int b)
        {
            return a + b;
        }

        public int Subtract(int a, int b)
        {
            return a - b;
        }

        public double Multiply(double x, double y)
        {
            return x * y;
        }
    }

    public interface IDataService
    {
        void SaveData(string data);
        string LoadData();
    }

    public class DataService : IDataService
    {
        public void SaveData(string data)
        {
            Console.WriteLine($"Saving: {data}");
        }

        public string LoadData()
        {
            return "Sample data";
        }
    }
}
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cs', delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def test_csharp_class_extraction(self, sample_csharp_file):
        """Test that C# classes are correctly extracted."""
        dependencies = get_csharp_dependencies(sample_csharp_file)
        
        assert dependencies is not None
        assert 'classes' in dependencies or len(dependencies) > 0
        
        # Check if Calculator class is detected
        classes = dependencies.get('classes', dependencies)
        class_names = [c.get('name', c) for c in classes] if isinstance(classes, list) else []
        
        assert 'Calculator' in str(dependencies)

    def test_csharp_function_extraction(self, sample_csharp_file):
        """Test that C# methods are correctly extracted."""
        dependencies = get_csharp_dependencies(sample_csharp_file)
        
        assert dependencies is not None
        
        # Check if methods are detected
        assert 'Add' in str(dependencies) or 'functions' in str(dependencies)

    def test_csharp_interface_extraction(self, sample_csharp_file):
        """Test that C# interfaces are correctly extracted."""
        dependencies = get_csharp_dependencies(sample_csharp_file)
        
        assert dependencies is not None
        assert 'IDataService' in str(dependencies)

    def test_csharp_using_statements(self, sample_csharp_file):
        """Test that C# using statements (imports) are detected."""
        dependencies = get_csharp_dependencies(sample_csharp_file)
        
        assert dependencies is not None
        # Check for imports/using statements
        dep_str = str(dependencies)
        assert 'System' in dep_str or 'using' in dep_str.lower() or 'imports' in dep_str.lower()


class TestCppDependencies:
    """Test suite for C++ dependency extraction."""

    @pytest.fixture
    def sample_cpp_file(self):
        """Create a temporary C++ file with sample code."""
        code = '''
#include <iostream>
#include <vector>
#include <string>
#include "custom_header.h"

namespace MathUtils {
    class Calculator {
    private:
        int result;
    
    public:
        Calculator() : result(0) {}
        
        int add(int a, int b) {
            return a + b;
        }
        
        int subtract(int a, int b) {
            return a - b;
        }
        
        double multiply(double x, double y) {
            return x * y;
        }
    };

    class ScientificCalculator : public Calculator {
    public:
        double power(double base, double exponent) {
            return std::pow(base, exponent);
        }
        
        double squareRoot(double value) {
            return std::sqrt(value);
        }
    };
}

void printResult(int value) {
    std::cout << "Result: " << value << std::endl;
}

int main() {
    MathUtils::Calculator calc;
    int sum = calc.add(5, 3);
    printResult(sum);
    return 0;
}
'''
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

    def test_cpp_class_extraction(self, sample_cpp_file):
        """Test that C++ classes are correctly extracted."""
        dependencies = get_cpp_dependencies(sample_cpp_file)
        
        assert dependencies is not None
        assert 'classes' in dependencies or len(dependencies) > 0
        
        # Check if Calculator class is detected
        assert 'Calculator' in str(dependencies)

    def test_cpp_function_extraction(self, sample_cpp_file):
        """Test that C++ functions are correctly extracted."""
        dependencies = get_cpp_dependencies(sample_cpp_file)
        
        assert dependencies is not None
        
        # Check if functions are detected (add, subtract, multiply, printResult, main)
        dep_str = str(dependencies)
        assert 'add' in dep_str or 'printResult' in dep_str or 'functions' in dep_str

    def test_cpp_namespace_extraction(self, sample_cpp_file):
        """Test that C++ namespaces are correctly extracted."""
        dependencies = get_cpp_dependencies(sample_cpp_file)
        
        assert dependencies is not None
        assert 'MathUtils' in str(dependencies) or 'namespace' in str(dependencies).lower()

    def test_cpp_include_statements(self, sample_cpp_file):
        """Test that C++ include statements are detected."""
        dependencies = get_cpp_dependencies(sample_cpp_file)
        
        assert dependencies is not None
        # Check for includes
        dep_str = str(dependencies)
        assert 'iostream' in dep_str or 'include' in dep_str.lower() or 'imports' in dep_str.lower()

    def test_cpp_inheritance(self, sample_cpp_file):
        """Test that C++ class inheritance is detected."""
        dependencies = get_cpp_dependencies(sample_cpp_file)
        
        assert dependencies is not None
        # Check for ScientificCalculator which inherits from Calculator
        assert 'ScientificCalculator' in str(dependencies)


class TestIntegration:
    """Integration tests for the codegraph pipeline."""

    def test_csharp_file_processing(self):
        """Test that .cs files are processed correctly."""
        code = 'namespace Test { public class Sample { public void Method() {} } }'
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cs', delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            result = get_csharp_dependencies(temp_path)
            assert result is not None
            assert 'Sample' in str(result)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_cpp_file_processing(self):
        """Test that .cpp files are processed correctly."""
        code = 'class Sample { public: void method() {} };'
        with tempfile.NamedTemporaryFile(mode='w', suffix='.cpp', delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            result = get_cpp_dependencies(temp_path)
            assert result is not None
            assert 'Sample' in str(result)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
