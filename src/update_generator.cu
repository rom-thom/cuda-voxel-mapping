#include "voxel-mapping/update_generator.cuh"
#include "voxel-mapping/host_macros.hpp"
#include "voxel-mapping/map_utils.cuh"
#include <limits.h>

static __constant__ float d_fx;
static __constant__ float d_fy;
static __constant__ float d_cx;
static __constant__ float d_cy;
static __constant__ uint32_t d_image_width;
static __constant__ uint32_t d_image_height;
static __constant__ float d_resolution;
static __constant__ float d_min_depth;
static __constant__ float d_max_depth;
static __constant__ uint32_t d_update_list_capacity;

UpdateGenerator::UpdateGenerator(size_t capacity, float voxel_resolution, float min_depth, float max_depth){
    CHECK_CUDA_ERROR(cudaMalloc(&d_update_list_, capacity * sizeof(VoxelUpdate)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_update_counter_, sizeof(uint32_t)));
    CHECK_CUDA_ERROR(cudaMemset(d_update_counter_, 0, sizeof(uint32_t)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_seen_voxel_counter_, sizeof(uint32_t)));
    CHECK_CUDA_ERROR(cudaMemset(d_seen_voxel_counter_, 0, sizeof(uint32_t)));

    
    VoxelKey empty_voxel_key_sentinel = std::numeric_limits<VoxelKey>::max();
    uint32_t empty_voxel_value_sentinel = 0u;
    
    d_voxel_update_set_ = std::make_unique<VoxelUpdateSet>(capacity,
        cuco::empty_key{empty_voxel_key_sentinel},
        cuco::empty_value{empty_voxel_value_sentinel}
    );
    CHECK_CUDA_ERROR(cudaMalloc(&d_transform_, 16 * sizeof(float)));

    uint32_t capacity_uint = static_cast<uint32_t>(capacity);
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_update_list_capacity, &capacity_uint, sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_resolution, &voxel_resolution, sizeof(float), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_min_depth, &min_depth, sizeof(float), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_max_depth, &max_depth, sizeof(float), 0, cudaMemcpyHostToDevice));
}

UpdateGenerator::~UpdateGenerator() {
    cudaFree(d_update_list_);
    cudaFree(d_update_counter_);
    cudaFree(d_seen_voxel_counter_);
    cudaFree(d_depth_);
    cudaFree(d_transform_);
}

void UpdateGenerator::set_camera_properties(
    float fx, float fy, float cx, float cy,
    uint32_t width, uint32_t height) {
    image_width_ = width;
    image_height_ = height;
    
    depth_buffer_size_ = static_cast<size_t>(width) * height * sizeof(float);
    
    CHECK_CUDA_ERROR(cudaMalloc(&d_depth_, depth_buffer_size_));

    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_fx, &fx, sizeof(float), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_fy, &fy, sizeof(float), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_cx, &cx, sizeof(float), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_cy, &cy, sizeof(float), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_image_width, &image_width_, sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_image_height, &image_height_, sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
}

void UpdateGenerator::reset_update_state(cudaStream_t stream) {
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_update_counter_, 0, sizeof(uint32_t), stream));
    CHECK_CUDA_ERROR(cudaMemsetAsync(d_seen_voxel_counter_, 0, sizeof(uint32_t), stream));
    d_voxel_update_set_->clear(stream);
}

uint32_t UpdateGenerator::generate_updates(
    const float* h_depth_image, 
    const float* transform,
    cudaStream_t stream) 
{
    CHECK_CUDA_ERROR(cudaMemcpy(d_transform_, transform, 16 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_depth_, h_depth_image, depth_buffer_size_, cudaMemcpyHostToDevice));

    reset_update_state(stream);
    launch_raycasting_kernel(stream);

    uint32_t num_updates;
    CHECK_CUDA_ERROR(cudaMemcpy(&num_updates, d_update_counter_, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    return num_updates;
}

__global__ void occupied_pass_kernel(
    const float* d_depth,
    const float* d_transform,
    UpdateSetRef seen_set_view,
    VoxelUpdate* update_list,
    uint32_t* d_update_counter,
    uint32_t* d_seen_voxel_counter)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= d_image_width || y >= d_image_height) return;

    float depth = d_depth[y * d_image_width + x];
    if (depth < d_min_depth || depth <= 0.0f) return;
    if (depth > d_max_depth) return;

    float cam_x = (x - d_cx) * depth / d_fx;
    float cam_y = (y - d_cy) * depth / d_fy;
    float cam_z = depth;
    float world_x = d_transform[0] * cam_x + d_transform[4] * cam_y + d_transform[8] * cam_z + d_transform[12];
    float world_y = d_transform[1] * cam_x + d_transform[5] * cam_y + d_transform[9] * cam_z + d_transform[13];
    float world_z = d_transform[2] * cam_x + d_transform[6] * cam_y + d_transform[10] * cam_z + d_transform[14];

    int ix = static_cast<int>(floor(world_x / d_resolution));
    int iy = static_cast<int>(floor(world_y / d_resolution));
    int iz = static_cast<int>(floor(world_z / d_resolution));

    ChunkKey key = get_chunk_key(ix, iy, iz);
    uint32_t idx = get_intra_chunk_index(ix, iy, iz);
    VoxelKey voxel_key = pack_indices_to_key(ix, iy, iz);

    VoxelUpdateSet::value_type pair = {voxel_key, voxel_flag_value()};
    auto result = seen_set_view.insert_and_find(pair);
    if (result.second) {
        uint32_t voxel_count = atomicAdd(d_seen_voxel_counter, 1);
        if (voxel_count < d_update_list_capacity) {
            uint32_t output_idx = atomicAdd(d_update_counter, 1);
            if (output_idx < d_update_list_capacity) {
                update_list[output_idx] = {key, idx, UpdateType::Occupied};
            } else {
                return;
            }
        } else {
            return;
        }
    }
}

__global__ void free_pass_kernel(
    const float* d_depth,
    const float* d_transform,
    UpdateSetRef seen_set_view,
    VoxelUpdate* update_list,
    uint32_t* d_update_counter,
    uint32_t* d_seen_voxel_counter)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= d_image_width || y >= d_image_height) return;

    float depth = d_depth[y * d_image_width + x];
    if (depth < d_min_depth || depth <= 0.0f) return;

    if (depth > d_max_depth) {
        depth = d_max_depth;
    }

    float cam_x = (x - d_cx) * depth / d_fx;
    float cam_y = (y - d_cy) * depth / d_fy;
    float cam_z = depth;
    float world_x = d_transform[0] * cam_x + d_transform[4] * cam_y + d_transform[8] * cam_z + d_transform[12];
    float world_y = d_transform[1] * cam_x + d_transform[5] * cam_y + d_transform[9] * cam_z + d_transform[13];
    float world_z = d_transform[2] * cam_x + d_transform[6] * cam_y + d_transform[10] * cam_z + d_transform[14];

    ChunkKey last_key = ULLONG_MAX;
    uint32_t last_idx = UINT_MAX;
    float start_x = d_transform[12];
    float start_y = d_transform[13];
    float start_z = d_transform[14];
    float step_size = d_resolution * 0.5f;
    float dx = world_x - start_x;
    float dy = world_y - start_y;
    float dz = world_z - start_z;
    float distance = sqrtf(dx * dx + dy * dy + dz * dz);
    uint32_t steps = static_cast<uint32_t>(distance / step_size);
    if (steps == 0) return;

    for (uint32_t i = 0; i < steps; ++i) {
        float t = static_cast<float>(i) / steps;

        float wx = start_x + t * dx;
        float wy = start_y + t * dy;
        float wz = start_z + t * dz;

        int ix = static_cast<int>(floor(wx / d_resolution));
        int iy = static_cast<int>(floor(wy / d_resolution));
        int iz = static_cast<int>(floor(wz / d_resolution));

        ChunkKey current_key = get_chunk_key(ix, iy, iz);
        uint32_t current_idx = get_intra_chunk_index(ix, iy, iz);

        if (current_key != last_key || current_idx != last_idx) {
            VoxelKey voxel_key = pack_indices_to_key(ix, iy, iz);

            auto found_it = seen_set_view.find(voxel_key);

            if (found_it == seen_set_view.end()) {
                VoxelUpdateSet::value_type pair = {voxel_key, voxel_flag_value()};
                auto result = seen_set_view.insert_and_find(pair);
                if (result.second) {
                    uint32_t voxel_count = atomicAdd(d_seen_voxel_counter, 1);
                    if (voxel_count < d_update_list_capacity) {
                        uint32_t output_idx = atomicAdd(d_update_counter, 1);
                        if (output_idx < d_update_list_capacity) {
                            update_list[output_idx] = {current_key, current_idx, UpdateType::Free};
                        } else {
                            return;
                        }
                    } else {
                        return;
                    }
                }
            }
            last_key = current_key;
            last_idx = current_idx;
        }
    }
}

void UpdateGenerator::launch_raycasting_kernel(cudaStream_t stream) 
{
    dim3 threads(16, 16);
    dim3 blocks((image_width_ + threads.x - 1) / threads.x, (image_height_ + threads.y - 1) / threads.y);
    auto voxel_update_set_ref = d_voxel_update_set_->ref(cuco::op::find, cuco::op::insert_and_find);

    occupied_pass_kernel<<<blocks, threads, 0, stream>>>(
        d_depth_,
        d_transform_,
        voxel_update_set_ref,
        d_update_list_,
        d_update_counter_,
        d_seen_voxel_counter_
    );

    CHECK_CUDA_ERROR(cudaGetLastError());

    free_pass_kernel<<<blocks, threads, 0, stream>>>(
        d_depth_,
        d_transform_,
        voxel_update_set_ref,
        d_update_list_,
        d_update_counter_,
        d_seen_voxel_counter_
    );
}