#ifndef MACROS_HPP
#define MACROS_HPP

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#define CHECK_CUDA_ERROR(err) do { \
    if ((err) != cudaSuccess) { \
        spdlog::error("CUDA Error: {} at {}:{}", cudaGetErrorString(err), __FILE__, __LINE__); \
        throw std::runtime_error("CUDA Error occurred. See logs for details."); \
    } \
} while(0)

#define AABB_INDEX(x, y, z, size_x, size_y, size_z) ((y) * (size_x) * (size_z) + (x) * (size_z) + (z))
#define VOXEL_INDEX(x, y, z, size_x, size_y, size_z) ((y) * (size_x) * (size_z) + (x) * (size_z) + (z))
#define SLICE_INDEX(x, y, size_x) ((y) * (size_x) + (x))

#endif // MACROS_HPP