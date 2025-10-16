#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include "custom_types.h"
#include "logger.h"

namespace MathUtils {
    /**
     * Basic calculator class for testing C++ dependency extraction
     */
    class Calculator {
    private:
        int result;
        std::shared_ptr<Logger> logger;

    public:
        Calculator(std::shared_ptr<Logger> log) : result(0), logger(log) {}

        virtual ~Calculator() {}

        /**
         * Add two integers
         */
        int add(int a, int b) {
            result = a + b;
            if (logger) {
                logger->log("Addition result: " + std::to_string(result));
            }
            return result;
        }

        /**
         * Subtract two integers
         */
        int subtract(int a, int b) {
            result = a - b;
            return result;
        }

        /**
         * Multiply two numbers
         */
        double multiply(double x, double y) {
            return x * y;
        }

        /**
         * Divide two numbers with error handling
         */
        double divide(double numerator, double denominator) {
            if (denominator == 0) {
                throw std::invalid_argument("Cannot divide by zero");
            }
            return numerator / denominator;
        }

        int getResult() const {
            return result;
        }

        void reset() {
            result = 0;
        }
    };

    /**
     * Scientific calculator with advanced operations
     */
    class ScientificCalculator : public Calculator {
    private:
        static constexpr double PI = 3.14159265358979323846;

    public:
        ScientificCalculator(std::shared_ptr<Logger> log) : Calculator(log) {}

        /**
         * Calculate power
         */
        double power(double base, double exponent) {
            return std::pow(base, exponent);
        }

        /**
         * Calculate square root
         */
        double squareRoot(double value) {
            if (value < 0) {
                throw std::invalid_argument("Cannot calculate square root of negative number");
            }
            return std::sqrt(value);
        }

        /**
         * Calculate logarithm
         */
        double logarithm(double value, double base = M_E) {
            if (value <= 0 || base <= 0 || base == 1) {
                throw std::invalid_argument("Invalid logarithm arguments");
            }
            return std::log(value) / std::log(base);
        }

        /**
         * Calculate sine
         */
        double sine(double angle) {
            return std::sin(angle);
        }

        /**
         * Calculate cosine
         */
        double cosine(double angle) {
            return std::cos(angle);
        }
    };

    /**
     * Statistical calculator for data analysis
     */
    class StatisticalCalculator {
    private:
        std::vector<double> data;

    public:
        void addValue(double value) {
            data.push_back(value);
        }

        void clearData() {
            data.clear();
        }

        double mean() const {
            if (data.empty()) {
                throw std::runtime_error("No data available");
            }
            double sum = 0.0;
            for (double val : data) {
                sum += val;
            }
            return sum / data.size();
        }

        double median() {
            if (data.empty()) {
                throw std::runtime_error("No data available");
            }
            std::vector<double> sorted = data;
            std::sort(sorted.begin(), sorted.end());
            size_t n = sorted.size();
            if (n % 2 == 0) {
                return (sorted[n/2 - 1] + sorted[n/2]) / 2.0;
            } else {
                return sorted[n/2];
            }
        }

        double variance() const {
            if (data.size() < 2) {
                throw std::runtime_error("Insufficient data for variance");
            }
            double m = mean();
            double sum = 0.0;
            for (double val : data) {
                double diff = val - m;
                sum += diff * diff;
            }
            return sum / (data.size() - 1);
        }

        double standardDeviation() const {
            return std::sqrt(variance());
        }
    };
}

namespace DataStructures {
    /**
     * Interface for data storage operations
     */
    template<typename T>
    class IDataService {
    public:
        virtual ~IDataService() {}
        virtual void save(const std::string& key, const T& value) = 0;
        virtual T load(const std::string& key) = 0;
        virtual bool remove(const std::string& key) = 0;
        virtual bool exists(const std::string& key) const = 0;
    };

    /**
     * In-memory implementation of data service
     */
    template<typename T>
    class MemoryDataService : public IDataService<T> {
    private:
        std::map<std::string, T> storage;
        std::shared_ptr<Logger> logger;

    public:
        MemoryDataService(std::shared_ptr<Logger> log) : logger(log) {}

        void save(const std::string& key, const T& value) override {
            storage[key] = value;
            if (logger) {
                logger->log("Saved data with key: " + key);
            }
        }

        T load(const std::string& key) override {
            auto it = storage.find(key);
            if (it == storage.end()) {
                throw std::runtime_error("Key not found: " + key);
            }
            return it->second;
        }

        bool remove(const std::string& key) override {
            return storage.erase(key) > 0;
        }

        bool exists(const std::string& key) const override {
            return storage.find(key) != storage.end();
        }

        size_t size() const {
            return storage.size();
        }
    };
}

namespace Utilities {
    /**
     * Static utility functions
     */
    class MathHelper {
    public:
        static bool isEven(int number) {
            return number % 2 == 0;
        }

        static bool isPrime(int number) {
            if (number <= 1) return false;
            if (number <= 3) return true;
            if (number % 2 == 0 || number % 3 == 0) return false;

            for (int i = 5; i * i <= number; i += 6) {
                if (number % i == 0 || number % (i + 2) == 0)
                    return false;
            }
            return true;
        }

        static std::vector<int> getFibonacci(int count) {
            std::vector<int> result;
            if (count <= 0) return result;

            int a = 0, b = 1;
            for (int i = 0; i < count; i++) {
                result.push_back(a);
                int temp = a;
                a = b;
                b = temp + b;
            }
            return result;
        }

        static int factorial(int n) {
            if (n < 0) {
                throw std::invalid_argument("Factorial not defined for negative numbers");
            }
            if (n == 0 || n == 1) return 1;
            return n * factorial(n - 1);
        }
    };

    /**
     * String utility functions
     */
    class StringHelper {
    public:
        static std::string toUpper(const std::string& str) {
            std::string result = str;
            std::transform(result.begin(), result.end(), result.begin(), ::toupper);
            return result;
        }

        static std::string toLower(const std::string& str) {
            std::string result = str;
            std::transform(result.begin(), result.end(), result.begin(), ::tolower);
            return result;
        }

        static bool startsWith(const std::string& str, const std::string& prefix) {
            return str.size() >= prefix.size() && 
                   str.compare(0, prefix.size(), prefix) == 0;
        }

        static bool endsWith(const std::string& str, const std::string& suffix) {
            return str.size() >= suffix.size() && 
                   str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
        }
    };
}

/**
 * Global utility functions
 */
void printResult(int value) {
    std::cout << "Result: " << value << std::endl;
}

void printResult(double value) {
    std::cout << "Result: " << value << std::endl;
}

void printError(const std::string& message) {
    std::cerr << "Error: " << message << std::endl;
}

/**
 * Main function demonstrating usage
 */
int main() {
    try {
        auto logger = std::make_shared<ConsoleLogger>();
        MathUtils::Calculator calc(logger);
        
        int sum = calc.add(5, 3);
        printResult(sum);
        
        MathUtils::ScientificCalculator sciCalc(logger);
        double sqrtResult = sciCalc.squareRoot(16.0);
        printResult(sqrtResult);
        
        MathUtils::StatisticalCalculator statCalc;
        statCalc.addValue(10.0);
        statCalc.addValue(20.0);
        statCalc.addValue(30.0);
        double avg = statCalc.mean();
        printResult(avg);
        
        std::cout << "Is 17 prime? " << (Utilities::MathHelper::isPrime(17) ? "Yes" : "No") << std::endl;
        
        return 0;
    } catch (const std::exception& ex) {
        printError(ex.what());
        return 1;
    }
}
