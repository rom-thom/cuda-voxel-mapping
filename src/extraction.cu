#include "macros.hpp"
#include <cfloat>

static __constant__ float d_occupancy_threshold;
static __constant__ float d_free_threshold;



extern "C" void set_extraction_grid_constants_d(uint grid_size_x, uint grid_size_y, uint grid_size_z, float resolution) {
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_grid_size_x, &grid_size_x, sizeof(uint), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_grid_size_y, &grid_size_y, sizeof(uint), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_grid_size_z, &grid_size_z, sizeof(uint), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_resolution, &resolution, sizeof(float), 0, cudaMemcpyHostToDevice));
}

extern "C" void set_extraction_depth_range_d(float min_depth, float max_depth) {
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_min_depth, &min_depth, sizeof(float), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_max_depth, &max_depth, sizeof(float), 0, cudaMemcpyHostToDevice));
}

extern "C" void set_extraction_log_odds_properties_d(float log_odds_occupied, float log_odds_free, float log_odds_min, float log_odds_max, float occupancy_threshold, float free_threshold) {
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_log_odds_occupied, &log_odds_occupied, sizeof(float), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_log_odds_free, &log_odds_free, sizeof(float), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_log_odds_min, &log_odds_min, sizeof(float), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_log_odds_max, &log_odds_max, sizeof(float), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_occupancy_threshold, &occupancy_threshold, sizeof(float), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_free_threshold, &free_threshold, sizeof(float), 0, cudaMemcpyHostToDevice));
}

__global__ void extract_2d_slice_kernel(
    const float* d_voxel_grid, float* d_slice,
    int min_x, int max_x, int min_y, int max_y, int min_z, int max_z) {
    
    int slice_x = blockIdx.x * blockDim.x + threadIdx.x;
    int slice_y = blockIdx.y * blockDim.y + threadIdx.y;
    int slice_size_x = max_x - min_x + 1;
    int slice_size_y = max_y - min_y + 1;

    if (slice_x >= slice_size_x || slice_y >= slice_size_y) return;
    
    int global_x = slice_x + min_x;
    int global_y = slice_y + min_y;
    
    if (global_x < 0 || global_x >= d_grid_size_x ||
        global_y < 0 || global_y >= d_grid_size_y) return;

    float state = -1.0f;  // Default: unknown
    for (int z = min_z; z <= max_z; ++z) {
        int grid_idx = VOXEL_INDEX(global_x, global_y, z, d_grid_size_x, d_grid_size_y, d_grid_size_z);
        float log_odds = d_voxel_grid[grid_idx];

        if (log_odds >= d_occupancy_threshold) {
            state = 1.0f;  // Occupied
            break;
        } else if (log_odds <= d_free_threshold) {
            state = 0.0f;  // Free
        }
    }

    if (state != -1.0f) {
        d_slice[SLICE_INDEX(slice_x, slice_y, slice_size_x)] = state;
    }
}

extern "C" void launch_extract_2d_slice_kernel(
    const float* d_voxel_grid, float* d_slice,
    int min_x, int max_x, int min_y, int max_y, int min_z, int max_z,
    cudaStream_t stream) {

    dim3 threads(16, 16);
    dim3 blocks((max_x - min_x + 15) / 16, (max_y - min_y + 15) / 16);
    extract_2d_slice_kernel<<<blocks, threads, 0, stream>>>(
        d_voxel_grid, d_slice,
        min_x, max_x, min_y, max_y, min_z, max_z);
}

__global__ void extract_binary_slice_kernel(
    const float* d_voxel_grid, float* d_slice,
    int min_x, int max_x, int min_y, int max_y, int min_z, int max_z) {
    
    int slice_x = blockIdx.x * blockDim.x + threadIdx.x;
    int slice_y = blockIdx.y * blockDim.y + threadIdx.y;
    int slice_size_x = max_x - min_x + 1;
    int slice_size_y = max_y - min_y + 1;

    if (slice_x >= slice_size_x || slice_y >= slice_size_y) return;
    
    int global_x = slice_x + min_x;
    int global_y = slice_y + min_y;
    
    if (global_x < 0 || global_x >= d_grid_size_x ||
        global_y < 0 || global_y >= d_grid_size_y) return;

    float state = FLT_MAX;  // Default: unknown
    for (int z = min_z; z <= max_z; ++z) {
        int grid_idx = VOXEL_INDEX(global_x, global_y, z, d_grid_size_x, d_grid_size_y, d_grid_size_z);
        float log_odds = d_voxel_grid[grid_idx];

        if (log_odds >= d_occupancy_threshold) {
            state = 0;  // Occupied
            break;
        }
    }
    if (state != FLT_MAX) {
        d_slice[SLICE_INDEX(slice_x, slice_y, slice_size_x)] = state;
    }
}

__global__ void initialize_float_kernel(float* arr, float value, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        arr[idx] = value;
    }
}

extern "C" void launch_initialize_float_kernel(
    float* arr, float value, size_t n) {

    int threads_per_block = 256;
    int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
    initialize_float_kernel<<<blocks_per_grid, threads_per_block>>>(arr, value, n);
    
    CHECK_CUDA_ERROR(cudaGetLastError());
}

extern "C" void launch_extract_binary_slice_kernel(
    const float* d_voxel_grid, float* d_slice,
    int min_x, int max_x, int min_y, int max_y, int min_z, int max_z,
    cudaStream_t stream) {

    dim3 threads(16, 16);
    dim3 blocks((max_x - min_x + 15) / 16, (max_y - min_y + 15) / 16);
    extract_binary_slice_kernel<<<blocks, threads, 0, stream>>>(
        d_voxel_grid, d_slice,
        min_x, max_x, min_y, max_y, min_z, max_z);
}

__global__ void extract_dilated_slice_kernel(
    const float* d_voxel_grid, float* d_slice,
    int min_x, int max_x, int min_y, int max_y, int min_z, int max_z, int dilation_size) {
    
    extern __shared__ float shared_voxel[];

    int slice_x = blockIdx.x * blockDim.x + threadIdx.x;
    int slice_y = blockIdx.y * blockDim.y + threadIdx.y;
    int slice_size_x = max_x - min_x + 1;
    int slice_size_y = max_y - min_y + 1;

    if (slice_x >= slice_size_x || slice_y >= slice_size_y) return;
    
    int global_x = slice_x + min_x;
    int global_y = slice_y + min_y;
    
    if (global_x < 0 || global_x >= d_grid_size_x ||
        global_y < 0 || global_y >= d_grid_size_y) return;

    for (int z = min_z; z <= max_z; ++z) {
        int grid_idx = VOXEL_INDEX(global_x, global_y, z, d_grid_size_x, d_grid_size_y, d_grid_size_z);
        float log_odds = d_voxel_grid[grid_idx];
        
        if (log_odds >= d_occupancy_threshold) {
            d_slice[SLICE_INDEX(slice_x, slice_y, slice_size_x)] = 1.0f;
            break;
        }
        else if (log_odds <= d_free_threshold) {
            d_slice[SLICE_INDEX(slice_x, slice_y, slice_size_x)] = 0.0f;
        }
    }
        
    __syncthreads();
    
    int shared_size_y = blockDim.y + dilation_size * 2;
                                
    #define SHARED_INDEX(x, y) ((x + dilation_size) * shared_size_y + (y + dilation_size))

    int block_x = threadIdx.x;
    int block_y = threadIdx.y;

    if (block_x < dilation_size && slice_x - dilation_size >= 0) {
        shared_voxel[SHARED_INDEX(block_x - dilation_size, block_y)] = d_slice[SLICE_INDEX(slice_x - dilation_size, slice_y, slice_size_x)];
    }
    if (block_x >= blockDim.x - dilation_size && slice_x + dilation_size < d_grid_size_x) {
        shared_voxel[SHARED_INDEX(block_x + dilation_size, block_y)] = d_slice[SLICE_INDEX(slice_x + dilation_size, slice_y, slice_size_x)];
    }
    if (block_y < dilation_size && slice_y - dilation_size >= 0) {
        shared_voxel[SHARED_INDEX(block_x, block_y - dilation_size)] = d_slice[SLICE_INDEX(slice_x, slice_y - dilation_size, slice_size_x)];
    }
    if (block_y >= blockDim.y - dilation_size && slice_y + dilation_size < d_grid_size_y) {
        shared_voxel[SHARED_INDEX(block_x, block_y + dilation_size)] = d_slice[SLICE_INDEX(slice_x, slice_y + dilation_size, slice_size_x)];
    }

    shared_voxel[SHARED_INDEX(block_x, block_y)] = d_slice[SLICE_INDEX(slice_x, slice_y, slice_size_x)];

    __syncthreads();

    if(shared_voxel[SHARED_INDEX(block_x, block_y)] == 1.0f) {
        int count = 0;
        count += shared_voxel[SHARED_INDEX(block_x - 1, block_y - 1)] == 1.0f;
        count += shared_voxel[SHARED_INDEX(block_x, block_y - 1)] == 1.0f;
        count += shared_voxel[SHARED_INDEX(block_x + 1, block_y - 1)] == 1.0f;
        count += shared_voxel[SHARED_INDEX(block_x - 1, block_y)] == 1.0f;
        count += shared_voxel[SHARED_INDEX(block_x + 1, block_y)] == 1.0f;
        count += shared_voxel[SHARED_INDEX(block_x - 1, block_y + 1)] == 1.0f;
        count += shared_voxel[SHARED_INDEX(block_x, block_y + 1)] == 1.0f;
        count += shared_voxel[SHARED_INDEX(block_x + 1, block_y + 1)] == 1.0f;
        if (count < 2) {
            shared_voxel[SHARED_INDEX(block_x, block_y)] = 0.0f;
        }
    }
    if(shared_voxel[SHARED_INDEX(block_x, block_y)] == 1.0f) {
        for (int i = -dilation_size; i <= dilation_size; ++i) {
            for (int j = -dilation_size; j <= dilation_size; ++j) {
                if (block_x + i < 0 || block_x + i >= blockDim.x || block_y + j < 0 || block_y + j >= blockDim.y) continue;
                shared_voxel[SHARED_INDEX(block_x + i, block_y + j)] = 1.0f;
            }    
        }
    }

    __syncthreads();

    if(block_x < dilation_size && slice_x - dilation_size >= 0) {
        if(shared_voxel[SHARED_INDEX(block_x - dilation_size, block_y)] == 1.0f) {
            for (int i = 0; i <= dilation_size; ++i) {
                for (int j = -dilation_size; j <= dilation_size; ++j) {
                    shared_voxel[SHARED_INDEX(block_x - dilation_size + i, block_y + j)] = 1.0f;
                }    
            }
        }
    }
    if(block_x >= blockDim.x - dilation_size && slice_x + dilation_size < d_grid_size_x) {
        if(shared_voxel[SHARED_INDEX(block_x + dilation_size, block_y)] == 1.0f) {
            for (int i = -dilation_size; i <= 0; ++i) {
                for (int j = -dilation_size; j <= dilation_size; ++j) {
                    shared_voxel[SHARED_INDEX(block_x + dilation_size + i, block_y + j)] = 1.0f;
                }    
            }
        }
    }
    if(block_y < dilation_size && slice_y - dilation_size >= 0) {
        if(shared_voxel[SHARED_INDEX(block_x, block_y - dilation_size)] == 1.0f) {
            for (int i = -dilation_size; i <= dilation_size; ++i) {
                for (int j = 0; j <= dilation_size; ++j) {
                    shared_voxel[SHARED_INDEX(block_x + i, block_y - dilation_size + j)] = 1.0f;
                }    
            }
        }
    }
    if(block_y >= blockDim.y - dilation_size && slice_y + dilation_size < d_grid_size_y) {
        if(shared_voxel[SHARED_INDEX(block_x, block_y + dilation_size)] == 1.0f) {
            for (int i = -dilation_size; i <= dilation_size; ++i) {
                for (int j = -dilation_size; j <= 0; ++j) {
                    shared_voxel[SHARED_INDEX(block_x + i, block_y + dilation_size + j)] = 1.0f;
                }    
            }
        }
    }

    __syncthreads();

    d_slice[SLICE_INDEX(slice_x, slice_y, slice_size_x)] = shared_voxel[SHARED_INDEX(block_x, block_y)];
}

extern "C" void launch_extract_dilated_2d_slice_kernel(
    const float* d_voxel_grid, float* d_slice,
    int min_x, int max_x, int min_y, int max_y, int min_z, int max_z, int dilation_size,
    cudaStream_t stream) {

    dim3 threads(16, 16);
    dim3 blocks((max_x - min_x + threads.x - 1) / threads.x, 
                (max_y - min_y + threads.y - 1) / threads.y);

    int shared_memory_size = (threads.x + dilation_size * 2) * (threads.y + dilation_size * 2) * sizeof(float);
    extract_dilated_slice_kernel<<<blocks, threads, shared_memory_size, stream>>>(
        d_voxel_grid, d_slice, min_x, max_x, min_y, max_y, min_z, max_z, dilation_size);
}

__global__ void edt_col_kernel(float* img, float* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;

    if (x >= height || y >= width) return;

    int rowsPerThread = (height + blockDim.x - 1) / blockDim.x;
    int untilPixel = min(x + rowsPerThread, height);

    extern __shared__ float img_col[];

    for (int row = threadIdx.x; row < height; row += blockDim.x) {
        img_col[row] = img[row * width + y];
    }
    __syncthreads();

    for (int row = x; row < untilPixel; row += blockDim.x) {
        float value = img_col[row];

        for (int row_i = 1, d = 1; row_i <= height - row - 1; row_i++) {
            if (row + row_i < height) {
                value = fminf(value, img_col[row + row_i] + d);
            }
            d += 1 + 2 * row_i;
        }

        for (int row_i = 1, d = 1; row_i <= row; row_i++) {
            if (row - row_i >= 0) {
                value = fminf(value, img_col[row - row_i] + d);
            }
            d += 1 + 2 * row_i;
        }

        out[row * width + y] = value;
    }
}

__global__ void edt_row_kernel(float* out, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * width;

    if (x >= width || y / width >= height) return;

    int colsPerThread = (width + blockDim.x - 1) / blockDim.x;
    int untilPixel = min(x + colsPerThread, width);

    extern __shared__ float imgRow[];

    for (int col = threadIdx.x; col < width; col += blockDim.x) {
        imgRow[col] = out[y + col];
    }
    __syncthreads();

    for (int col = x; col < untilPixel; col += blockDim.x) {
        float value = imgRow[col];

        for (int col_i = 1, d = 1; col_i <= width - col - 1; col_i++) {
            if (col + col_i < width) {
                value = fminf(value, imgRow[col + col_i] + d);
            }
            d += 1 + 2 * col_i;
        }

        for (int col_i = 1, d = 1; col_i <= col; col_i++) {
            if (col - col_i >= 0) {
                value = fminf(value, imgRow[col - col_i] + d);
            }
            d += 1 + 2 * col_i;
        }

        out[y + col] = value;
    }
}

extern "C" void launch_edt_kernels(float* d_binary_slice, float* d_edt, int width, int height, cudaStream_t stream) {
    int threadsPerBlock = 256;

    dim3 blockDim(threadsPerBlock, 1);
    dim3 gridDim((height + threadsPerBlock - 1) / threadsPerBlock, width);
    edt_col_kernel<<<gridDim, blockDim, height * sizeof(float), stream>>>(d_binary_slice, d_edt, width, height);

    CHECK_CUDA_ERROR(cudaGetLastError());

    dim3 gridDim2((width + threadsPerBlock - 1) / threadsPerBlock, height);
    edt_row_kernel<<<gridDim2, blockDim, width * sizeof(float), stream>>>(d_edt, width, height);

    CHECK_CUDA_ERROR(cudaGetLastError());
}