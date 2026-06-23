#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

std::size_t parse_size(const char* text) {
    const std::string value(text);
    std::size_t parsed_chars = 0;
    const unsigned long long parsed = std::stoull(value, &parsed_chars);
    if (parsed_chars != value.size() || parsed == 0 ||
        parsed > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("array size must be a positive integer");
    }
    return static_cast<std::size_t>(parsed);
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const std::size_t n = (argc >= 2) ? parse_size(argv[1]) : 1'000'000;

        std::vector<float> a(n);
        std::vector<float> b(n);
        std::vector<float> c(n, 0.0f);

        for (std::size_t i = 0; i < n; ++i) {
            a[i] = static_cast<float>(i) * 0.5f;
            b[i] = static_cast<float>(i) * 0.25f;
        }

        // CPU executes the additions one element after another.
        const auto start = std::chrono::high_resolution_clock::now();
        for (std::size_t i = 0; i < n; ++i) {
            c[i] = a[i] + b[i];
        }
        const auto stop = std::chrono::high_resolution_clock::now();

        bool passed = true;
        for (std::size_t i = 0; i < n; ++i) {
            const float expected = a[i] + b[i];
            if (std::fabs(c[i] - expected) > 1.0e-5f) {
                std::cerr << "Mismatch at index " << i << ": got " << c[i]
                          << ", expected " << expected << '\n';
                passed = false;
                break;
            }
        }

        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(stop - start).count();

        std::cout << std::fixed << std::setprecision(6)
                  << "RESULT mode=CPU"
                  << " n=" << n
                  << " time_ms=" << elapsed_ms
                  << " verification=" << (passed ? "PASS" : "FAIL") << '\n';
        return passed ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "Usage: vector_add_cpu [positive_array_size]\n"
                  << "Error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
