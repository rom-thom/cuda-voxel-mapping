#ifndef KERNEL_LAUNCHES_HPP
#define KERNEL_LAUNCHES_HPP

#include <cuda_runtime.h>

extern "C" void launch_process_depth_kernels(
    const float* d_depth, int width, int height,
    const float* d_transform, float* d_voxel_grid, char* d_aabb,
    int min_x, int max_x, int min_y, int max_y, int min_z, int max_z,
    cudaStream_t stream);

extern "C" void launch_extract_2d_slice_kernel(
    const float* d_voxel_grid, float* d_slice,
    int min_x, int max_x, int min_y, int max_y, int min_z, int max_z, cudaStream_t stream);


extern "C" void launch_extract_dilated_2d_slice_kernel(
    const float* d_voxel_grid, float* d_slice,
    int min_x, int max_x, int min_y, int max_y, int min_z, int max_z, int dilation_size,
    cudaStream_t stream);

extern "C" void launch_extract_binary_slice_kernel(
    const float* d_voxel_grid, float* d_slice,
    int min_x, int max_x, int min_y, int max_y, int min_z, int max_z,
    cudaStream_t stream);

extern "C" void launch_edt_kernels(
    float* d_binary_slice, float* d_edt,
    int width, int height,
    cudaStream_t stream);
    
extern "C" void launch_initialize_float_kernel(
    float* arr, float value, size_t n);

#endif // KERNEL_LAUNCHES_HPP