using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using MyCompany.Models;
using MyCompany.Interfaces;

namespace MyCompany.Services
{
    /// <summary>
    /// Sample calculator class for testing C# dependency extraction
    /// </summary>
    public class Calculator
    {
        private int _result;
        private readonly ILogger _logger;

        public Calculator(ILogger logger)
        {
            _result = 0;
            _logger = logger;
        }

        /// <summary>
        /// Adds two integers
        /// </summary>
        public int Add(int a, int b)
        {
            _result = a + b;
            _logger.Log($"Addition result: {_result}");
            return _result;
        }

        /// <summary>
        /// Subtracts two integers
        /// </summary>
        public int Subtract(int a, int b)
        {
            _result = a - b;
            return _result;
        }

        /// <summary>
        /// Multiplies two numbers
        /// </summary>
        public double Multiply(double x, double y)
        {
            return x * y;
        }

        /// <summary>
        /// Divides two numbers with error handling
        /// </summary>
        public double Divide(double numerator, double denominator)
        {
            if (denominator == 0)
            {
                throw new DivideByZeroException("Cannot divide by zero");
            }
            return numerator / denominator;
        }

        public async Task<int> AddAsync(int a, int b)
        {
            await Task.Delay(100);
            return Add(a, b);
        }
    }

    /// <summary>
    /// Scientific calculator with advanced operations
    /// </summary>
    public class ScientificCalculator : Calculator
    {
        public ScientificCalculator(ILogger logger) : base(logger)
        {
        }

        public double Power(double baseNum, double exponent)
        {
            return Math.Pow(baseNum, exponent);
        }

        public double SquareRoot(double value)
        {
            if (value < 0)
            {
                throw new ArgumentException("Cannot calculate square root of negative number");
            }
            return Math.Sqrt(value);
        }

        public double Logarithm(double value, double baseNum = Math.E)
        {
            return Math.Log(value, baseNum);
        }
    }

    /// <summary>
    /// Interface for data storage operations
    /// </summary>
    public interface IDataService
    {
        void SaveData(string key, object value);
        T LoadData<T>(string key);
        bool DeleteData(string key);
        Task<bool> SaveDataAsync(string key, object value);
    }

    /// <summary>
    /// Implementation of data service using in-memory storage
    /// </summary>
    public class MemoryDataService : IDataService
    {
        private readonly Dictionary<string, object> _storage;
        private readonly ILogger _logger;

        public MemoryDataService(ILogger logger)
        {
            _storage = new Dictionary<string, object>();
            _logger = logger;
        }

        public void SaveData(string key, object value)
        {
            _storage[key] = value;
            _logger.Log($"Saved data with key: {key}");
        }

        public T LoadData<T>(string key)
        {
            if (_storage.TryGetValue(key, out var value))
            {
                return (T)value;
            }
            throw new KeyNotFoundException($"Key not found: {key}");
        }

        public bool DeleteData(string key)
        {
            return _storage.Remove(key);
        }

        public async Task<bool> SaveDataAsync(string key, object value)
        {
            await Task.Run(() => SaveData(key, value));
            return true;
        }
    }

    /// <summary>
    /// Logger interface for dependency injection
    /// </summary>
    public interface ILogger
    {
        void Log(string message);
        void LogError(string message, Exception ex);
    }

    /// <summary>
    /// Console logger implementation
    /// </summary>
    public class ConsoleLogger : ILogger
    {
        public void Log(string message)
        {
            Console.WriteLine($"[INFO] {DateTime.Now}: {message}");
        }

        public void LogError(string message, Exception ex)
        {
            Console.WriteLine($"[ERROR] {DateTime.Now}: {message} - {ex.Message}");
        }
    }

    /// <summary>
    /// Static utility class for common operations
    /// </summary>
    public static class MathUtilities
    {
        public static bool IsEven(int number)
        {
            return number % 2 == 0;
        }

        public static bool IsPrime(int number)
        {
            if (number <= 1) return false;
            if (number <= 3) return true;
            
            for (int i = 2; i <= Math.Sqrt(number); i++)
            {
                if (number % i == 0) return false;
            }
            return true;
        }

        public static IEnumerable<int> GetFibonacci(int count)
        {
            int a = 0, b = 1;
            for (int i = 0; i < count; i++)
            {
                yield return a;
                int temp = a;
                a = b;
                b = temp + b;
            }
        }
    }
}
