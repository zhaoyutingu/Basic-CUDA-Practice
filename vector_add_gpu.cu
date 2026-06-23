#include <cuda_runtime.h>

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

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        const cudaError_t error_code = (call);                                 \
        if (error_code != cudaSuccess) {                                       \
            std::cerr << "CUDA error at " << __FILE__ << ':' << __LINE__       \
                      << ": " << cudaGetErrorString(error_code) << '\n';       \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (false)

__global__ void vector_add(const float* a, const float* b, float* c,
                           std::size_t n) {
    // 每个 thread 根据自己的全局编号 idx 计算一个数组元素。
    const std::size_t idx =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // 最后一个 block 可能没有被完全用满，因此要防止越界访问。
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

namespace {

unsigned long long parse_positive(const char* text, const char* name) {
    const std::string value(text);
    std::size_t parsed_chars = 0;
    const unsigned long long parsed = std::stoull(value, &parsed_chars);
    if (parsed_chars != value.size() || parsed == 0) {
        throw std::invalid_argument(std::string(name) +
                                    " must be a positive integer");
    }
    return parsed;
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        const unsigned long long n_input =
            (argc >= 2) ? parse_positive(argv[1], "array size") : 1'000'000ULL;
        const unsigned int block_size = static_cast<unsigned int>(
            (argc >= 3) ? parse_positive(argv[2], "block size") : 256ULL);
        const int repeats = static_cast<int>(
            (argc >= 4) ? parse_positive(argv[3], "repeat count") : 10ULL);

        if (n_input > std::numeric_limits<std::size_t>::max()) {
            throw std::invalid_argument("array size is too large for this system");
        }
        if (block_size == 0 || block_size > 1024) {
            throw std::invalid_argument("block size must be between 1 and 1024");
        }
        if (repeats <= 0) {
            throw std::invalid_argument("repeat count is too large");
        }

        const std::size_t n = static_cast<std::size_t>(n_input);
        const std::size_t bytes = n * sizeof(float);
        const unsigned long long grid_size_ull =
            (n_input + block_size - 1ULL) / block_size;
        if (grid_size_ull > std::numeric_limits<unsigned int>::max()) {
            throw std::invalid_argument("grid size exceeds CUDA's one-dimensional limit");
        }
        const unsigned int grid_size = static_cast<unsigned int>(grid_size_ull);

        std::vector<float> host_a(n);
        std::vector<float> host_b(n);
        std::vector<float> host_c(n, 0.0f);
        for (std::size_t i = 0; i < n; ++i) {
            host_a[i] = static_cast<float>(i) * 0.5f;
            host_b[i] = static_cast<float>(i) * 0.25f;
        }

        float* device_a = nullptr;
        float* device_b = nullptr;
        float* device_c = nullptr;
        cudaEvent_t kernel_start;
        cudaEvent_t kernel_stop;

        const auto total_start = std::chrono::high_resolution_clock::now();
        CUDA_CHECK(cudaMalloc(&device_a, bytes));
        CUDA_CHECK(cudaMalloc(&device_b, bytes));
        CUDA_CHECK(cudaMalloc(&device_c, bytes));
        CUDA_CHECK(cudaMemcpy(device_a, host_a.data(), bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_b, host_b.data(), bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventCreate(&kernel_start));
        CUDA_CHECK(cudaEventCreate(&kernel_stop));

        // 预热一次，减少 CUDA 上下文初始化对 kernel 计时的影响。
        vector_add<<<grid_size, block_size>>>(device_a, device_b, device_c, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(kernel_start));
        for (int run = 0; run < repeats; ++run) {
            vector_add<<<grid_size, block_size>>>(device_a, device_b, device_c, n);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(kernel_stop));
        CUDA_CHECK(cudaEventSynchronize(kernel_stop));

        float all_kernel_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&all_kernel_ms, kernel_start, kernel_stop));
        CUDA_CHECK(cudaMemcpy(host_c.data(), device_c, bytes,
                              cudaMemcpyDeviceToHost));
        const auto total_stop = std::chrono::high_resolution_clock::now();

        bool passed = true;
        for (std::size_t i = 0; i < n; ++i) {
            const float expected = host_a[i] + host_b[i];
            if (std::fabs(host_c[i] - expected) > 1.0e-5f) {
                std::cerr << "Mismatch at index " << i << ": got " << host_c[i]
                          << ", expected " << expected << '\n';
                passed = false;
                break;
            }
        }

        CUDA_CHECK(cudaEventDestroy(kernel_start));
        CUDA_CHECK(cudaEventDestroy(kernel_stop));
        CUDA_CHECK(cudaFree(device_a));
        CUDA_CHECK(cudaFree(device_b));
        CUDA_CHECK(cudaFree(device_c));

        const double total_ms =
            std::chrono::duration<double, std::milli>(total_stop - total_start)
                .count();
        const double average_kernel_ms = all_kernel_ms / repeats;

        std::cout << std::fixed << std::setprecision(6)
                  << "RESULT mode=GPU"
                  << " n=" << n
                  << " block_size=" << block_size
                  << " grid_size=" << grid_size
                  << " repeats=" << repeats
                  << " kernel_ms=" << average_kernel_ms
                  << " total_ms=" << total_ms
                  << " verification=" << (passed ? "PASS" : "FAIL") << '\n';
        return passed ? EXIT_SUCCESS : EXIT_FAILURE;
    } catch (const std::exception& error) {
        std::cerr << "Usage: vector_add_gpu [positive_array_size] "
                     "[block_size_1_to_1024] [positive_repeat_count]\n"
                  << "Error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
