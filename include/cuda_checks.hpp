#ifndef CUDA_CHECKS_HPP
#define CUDA_CHECKS_HPP

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#define CHECK_CUDA_ERROR(err) do { \
    if ((err) != cudaSuccess) { \
        spdlog::error("CUDA Error: {} at {}:{}", cudaGetErrorString(err), __FILE__, __LINE__); \
        throw std::runtime_error("CUDA Error occurred. See logs for details."); \
    } \
} while(0)

#endif // CUDA_CHECKS_HPP