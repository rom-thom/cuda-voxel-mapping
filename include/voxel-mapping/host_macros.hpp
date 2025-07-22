#ifndef HOST_MACROS_HPP
#define HOST_MACROS_HPP

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#define CHECK_CUDA_ERROR(err) do { \
    const cudaError_t err_code = (err); \
    if (err_code != cudaSuccess) { \
        spdlog::error("CUDA Error: {} at {}:{}", cudaGetErrorString(err_code), __FILE__, __LINE__); \
        throw std::runtime_error("CUDA Error occurred. See logs for details."); \
    } \
} while(0)

#endif // HOST_MACROS_HPP