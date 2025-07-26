#include "voxel-mapping/grid_processor.cuh"
#include "voxel-mapping/types.hpp"
#include "voxel-mapping/host_macros.hpp"
#include "voxel-mapping/map_utils.cuh"
#include <cuda_runtime.h>

namespace voxel_mapping {

static __constant__ VoxelType d_occupancy_threshold;
static __constant__ VoxelType d_free_threshold;

GridProcessor::GridProcessor(int occupancy_threshold, int free_threshold)
    : occupancy_threshold_(occupancy_threshold), free_threshold_(free_threshold) {
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_occupancy_threshold, &occupancy_threshold_, sizeof(VoxelType), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_free_threshold, &free_threshold_, sizeof(VoxelType), 0, cudaMemcpyHostToDevice));
}

__global__ void extract_binary_slice_kernel(
    const VoxelType* d_aabb, int* d_slice,
    int min_x, int min_y, int min_z, int size_x, int size_y, int size_z) {
    
    int slice_x = blockIdx.x * blockDim.x + threadIdx.x;
    int slice_y = blockIdx.y * blockDim.y + threadIdx.y;
    int slice_size_x = size_x;
    int slice_size_y = size_y;

    if (slice_x >= slice_size_x || slice_y >= slice_size_y) return;

    int state = size_x > size_y ? size_x : size_y;
    for (int z = 0; z < size_z; ++z) {
        int aabb_idx = block_1d_index(slice_x, slice_y, z, size_x, size_y);
        VoxelType log_odds = d_aabb[aabb_idx];

        if (log_odds >= d_occupancy_threshold) {
            state = 0;
            break;
        }
    }

    d_slice[block_1d_index(slice_x, slice_y, 0, size_x, size_y)] = state;
}

void GridProcessor::launch_extract_binary_slice_kernel(
    const VoxelType* d_aabb, int* d_slice,
    int min_x, int min_y,int min_z, 
    int size_x, int size_y, int size_z,
    cudaStream_t stream) {

    dim3 threads(16, 16);
    dim3 blocks((size_x + 15) / 16, (size_y + 15) / 16);
    extract_binary_slice_kernel<<<blocks, threads, 0, stream>>>(
        d_aabb, d_slice,
        min_x, min_y, min_z, size_x, size_y, size_z);
}


__global__ void edt_col_kernel(int* input, int* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;

    if (x >= height || y >= width) return;

    int rows_per_thread = (height + blockDim.x - 1) / blockDim.x; // Calculate how many rows each thread should handle
    int until_row = min(x + rows_per_thread, height); // Calculate the last row this thread will handle

    extern __shared__ int col[];

    for (int row = threadIdx.x; row < height; row += blockDim.x) { // Load the column data into shared memory
        col[row] = input[row * width + y]; // Each block loads one block of rows for each iteration
    }
    __syncthreads();

    for (int row = x; row < until_row; row += blockDim.x) { // Iterate blockwise over rows
        int value = col[row]; // Initialize the value with the current pixel

        // Forward pass, iterate over the rows below the current row
        for (int row_i = 1, d = 1; row_i <= height - row - 1; row_i++) {
            if (row + row_i < height) {
                value = min(value, col[row + row_i] + d);
            }
            d += 1 + 2 * row_i;
        }

        // Backward pass, iterate over the rows above the current row
        for (int row_i = 1, d = 1; row_i <= row; row_i++) {
            if (row - row_i >= 0) {
                value = min(value, col[row - row_i] + d);
            }
            d += 1 + 2 * row_i;
        }

        out[row * width + y] = value;
    }
}

__global__ void edt_row_kernel(int* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;

    if (x >= width || y >= height) return;

    int cols_per_thread = (width + blockDim.x - 1) / blockDim.x;
    int until_col = min(x + cols_per_thread, width);

    extern __shared__ int row[];

    for (int col = threadIdx.x; col < width; col += blockDim.x) {
        row[col] = out[y * width + col];
    }

    __syncthreads();

    for (int col = x; col < until_col; col += blockDim.x) {
        int value = row[col];

        for (int col_i = 1, d = 1; col_i <= width - col - 1; col_i++) {
            if (col + col_i < width) {
                value = min(value, row[col + col_i] + d);
            }
            d += 1 + 2 * col_i;
        }

        for (int col_i = 1, d = 1; col_i <= col; col_i++) {
            if (col - col_i >= 0) {
                value = min(value, row[col - col_i] + d);
            }
            d += 1 + 2 * col_i;
        }

        out[y * width + col] = value;
    }
}

void GridProcessor::launch_edt_kernels(int* d_binary_slice, int* d_edt, int size_x, int size_y, cudaStream_t stream) {
    int threadsPerBlock = 256;

    dim3 blockDim(threadsPerBlock, 1);
    dim3 gridDim((size_y + threadsPerBlock - 1) / threadsPerBlock, size_x);
    edt_col_kernel<<<gridDim, blockDim, size_y * sizeof(int), stream>>>(d_binary_slice, d_edt, size_x, size_y);

    CHECK_CUDA_ERROR(cudaGetLastError());

    dim3 gridDim2((size_x + threadsPerBlock - 1) / threadsPerBlock, size_y);
    edt_row_kernel<<<gridDim2, blockDim, size_x * sizeof(int), stream>>>(d_edt, size_x, size_y);

    CHECK_CUDA_ERROR(cudaGetLastError());
}

} // namespace voxel_mapping
