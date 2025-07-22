#include "voxel-mapping/gpu_hash_map.cuh"
#include "voxel-mapping/host_macros.hpp"
#include "voxel-mapping/map_utils.cuh"
#include "voxel-mapping/internal_types.cuh"
#include <algorithm>
#include <vector>
#include <cstdint>
#include <limits>

static __constant__ VoxelType d_log_odds_occupied;
static __constant__ VoxelType d_log_odds_free;
static __constant__ VoxelType d_log_odds_min;
static __constant__ VoxelType d_log_odds_max;
static __constant__ VoxelType d_occupancy_threshold;
static __constant__ VoxelType d_free_threshold;

GpuHashMap::GpuHashMap(size_t capacity, VoxelType log_odds_occupied, VoxelType log_odds_free, VoxelType log_odds_min, VoxelType log_odds_max, VoxelType occupancy_threshold, VoxelType free_threshold) {
    if (capacity == 0) {
        spdlog::error("Capacity must be greater than zero.");
        throw std::invalid_argument("Capacity must be greater than zero.");
    }
    if (capacity > std::numeric_limits<uint32_t>::max()) {
        spdlog::error("Capacity exceeds the limit of the internal counter.");
        throw std::invalid_argument("Capacity exceeds the limit of the internal counter.");
    }

    map_capacity_ = capacity;

    ChunkKey empty_key_sentinel = std::numeric_limits<ChunkKey>::max();
    ChunkPtr empty_value_sentinel = nullptr;

    d_voxel_map_ = std::make_unique<ChunkMap>(
        capacity,
        cuco::empty_key{empty_key_sentinel},
        cuco::empty_value{empty_value_sentinel}
    );

    size_t chunk_size_elements = chunk_dim() * chunk_dim() * chunk_dim();
    size_t chunk_size_bytes = sizeof(VoxelType) * chunk_size_elements;
    size_t pool_size_bytes = chunk_size_bytes * map_capacity_;

    CHECK_CUDA_ERROR(cudaMalloc(&global_memory_pool_, pool_size_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&freelist_, sizeof(ChunkPtr) * map_capacity_));
    CHECK_CUDA_ERROR(cudaMalloc(&freelist_counter_, sizeof(uint32_t)));

    CHECK_CUDA_ERROR(cudaMemset(global_memory_pool_, default_voxel_value(), pool_size_bytes));

    std::vector<ChunkPtr> h_chunk_ptrs(map_capacity_);
    for (size_t i = 0; i < map_capacity_; ++i) {
        h_chunk_ptrs[i] = global_memory_pool_ + (i * chunk_size_elements);
    }

    CHECK_CUDA_ERROR(cudaMemcpy(
        freelist_,
        h_chunk_ptrs.data(),
        sizeof(ChunkPtr) * map_capacity_,
        cudaMemcpyHostToDevice
    ));

    uint32_t initial_count = 0;
    CHECK_CUDA_ERROR(cudaMemcpy(
        freelist_counter_,
        &initial_count,
        sizeof(uint32_t),
        cudaMemcpyHostToDevice
    ));

    CHECK_CUDA_ERROR(cudaMalloc(&freelist_allocation_counter_, sizeof(uint32_t)));
    uint32_t initial_allocation_count = 0;
    CHECK_CUDA_ERROR(cudaMemcpy(
        freelist_allocation_counter_, 
        &initial_allocation_count, 
        sizeof(uint32_t), 
        cudaMemcpyHostToDevice
    ));

    CHECK_CUDA_ERROR(cudaMalloc(&freelist_capacity_, sizeof(uint32_t)));
    uint32_t initial_capacity = static_cast<uint32_t>(map_capacity_);
    CHECK_CUDA_ERROR(cudaMemcpy(
        freelist_capacity_,
        &initial_capacity,
        sizeof(uint32_t),
        cudaMemcpyHostToDevice
    ));

    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_log_odds_occupied, &log_odds_occupied, sizeof(VoxelType), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_log_odds_free, &log_odds_free, sizeof(VoxelType), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_log_odds_min, &log_odds_min, sizeof(VoxelType), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_log_odds_max, &log_odds_max, sizeof(VoxelType), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_occupancy_threshold, &occupancy_threshold, sizeof(VoxelType), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_free_threshold, &free_threshold, sizeof(VoxelType), 0, cudaMemcpyHostToDevice));
}

GpuHashMap::~GpuHashMap() = default;

ChunkMap* GpuHashMap::get_map() {
    return d_voxel_map_.get();
}

// __device__ inline ChunkPtr allocate_chunk_from_pool(ChunkPtr* freelist, uint32_t* counter) {
//     uint32_t expected = *((volatile uint32_t*)counter);
    
//     while (expected > 0) {
//         uint32_t found = atomicCAS(counter, expected, expected - 1);
        
//         if (found == expected) {
//             return freelist[expected - 1];
//         }
//         expected = found;
//     }

//     return nullptr;
// }

__device__ inline ChunkPtr allocate_chunk_from_pool(ChunkPtr* freelist, uint32_t* counter, uint32_t* allocation_counter, uint32_t capacity) {
    uint32_t current_allocation = atomicAdd(allocation_counter, 1);
    if (current_allocation < capacity) {
        uint32_t current_index = atomicAdd(counter, 1);
        if (current_index < capacity) {
            return freelist[current_index];
        }
    }
    return nullptr;
}

__device__ inline void deallocate_chunk_to_pool(ChunkPtr chunk_ptr, ChunkPtr* freelist, uint32_t* counter) {
    uint32_t index = atomicSub(counter, 1);
    freelist[index] = chunk_ptr;
}


__device__ inline void atomic_add_and_clamp(VoxelType* address, VoxelType val, VoxelType min_val, VoxelType max_val) {
    VoxelType old_val = *address;
    VoxelType assumed_val;

    do {
        assumed_val = old_val;
        VoxelType new_val = min(max(assumed_val + val, min_val), max_val);
        old_val = atomicCAS(address, assumed_val, new_val);
    } while (assumed_val != old_val);
}

__device__ inline void update_voxels(ChunkPtr chunk_ptr, uint32_t intra_chunk_idx, UpdateType update_type) {
    VoxelType update_value = (update_type == UpdateType::Free) ? d_log_odds_free : d_log_odds_occupied;
    atomic_add_and_clamp(
        &chunk_ptr[intra_chunk_idx],
        update_value,
        d_log_odds_min,
        d_log_odds_max
    );
}
__global__ void update_map_from_keys_kernel(
    ChunkMapRef map_ref,
    ChunkPtr* freelist,
    uint32_t* freelist_counter,
    const VoxelUpdate* update_list,
    uint32_t num_updates,
    uint32_t* freelist_allocation_counter,
    uint32_t* freelist_capacity)
{
    // Determine the unique thread ID
    uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id >= num_updates) return;

    // Create a tiled cooperative group (warp-sized)
    auto group = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());

    VoxelUpdate my_update = update_list[thread_id];
    ChunkKey my_key = my_update.key;

    unsigned int active_mask = group.ballot(true);
    while (active_mask != 0) {
        int leader_lane = __ffs(active_mask) - 1;
        ChunkKey current_key = group.shfl(my_key, leader_lane);

        ChunkPtr final_chunk_ptr = invalid_chunk_ptr();

        auto it = map_ref.find(group, current_key);
        group.sync();

        if (it != map_ref.end()) {
            final_chunk_ptr = it->second;

        } else {
            ChunkMap::value_type pair_to_insert{current_key, invalid_chunk_ptr()};
            auto result = map_ref.insert_and_find(group, pair_to_insert);

            bool was_inserted = result.second;
            auto insert_it = result.first;

            if (was_inserted) {
                if (group.thread_rank() == leader_lane) {
                    ChunkPtr new_chunk_ptr = allocate_chunk_from_pool(freelist, freelist_counter, freelist_allocation_counter, *freelist_capacity);

                    if (new_chunk_ptr != nullptr) {
                        auto ref = cuda::atomic_ref<ChunkPtr, cuda::thread_scope_device>{insert_it->second};
    
                    ref.exchange(new_chunk_ptr, cuda::memory_order_release);
    
    
                        final_chunk_ptr = new_chunk_ptr;
                    }

            }
            final_chunk_ptr = group.shfl(final_chunk_ptr, leader_lane);
            } else {
                final_chunk_ptr = insert_it->second;
            }
        }

        if (my_key == current_key && final_chunk_ptr != invalid_chunk_ptr()) {
            update_voxels(final_chunk_ptr, my_update.intra_chunk_idx, my_update.update_type);
        }

        unsigned int processed_mask = group.ballot(my_key == current_key && final_chunk_ptr != invalid_chunk_ptr());
        active_mask &= ~processed_mask;
    }
}

void GpuHashMap::launch_map_update_kernel(
    uint32_t num_updates,
    const VoxelUpdate* update_list,
    cudaStream_t stream)
{
    auto map_ref = d_voxel_map_->ref(cuco::op::find, cuco::op::insert_and_find);

    uint32_t threads_per_block = 256;
    uint32_t blocks = (num_updates + threads_per_block - 1) / threads_per_block;

    update_map_from_keys_kernel<<<blocks, threads_per_block, 0, stream>>>(
        map_ref,
        freelist_,
        freelist_counter_,
        update_list,
        num_updates,
        freelist_allocation_counter_,
        freelist_capacity_

    );
}

__global__ void extract_block_kernel(
    ConstChunkMapRef map_ref,
    VoxelType* d_output_block,
    int min_x, int min_y, int min_z,
    size_t size_x, size_t size_y, size_t size_z)
{
    auto group = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());

    uint32_t aabb_x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t aabb_y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t aabb_z = blockIdx.z;

    if (aabb_x >= size_x || aabb_y >= size_y || aabb_z >= size_z) {
        return;
    }

    int global_x = min_x + aabb_x;
    int global_y = min_y + aabb_y;
    int global_z = min_z + aabb_z;

    ChunkKey my_key = get_chunk_key(global_x, global_y, global_z);

    uint32_t output_idx = block_1d_index(aabb_x, aabb_y, aabb_z, size_x, size_y);
    
    unsigned int active_mask = group.ballot(true);
    while(active_mask != 0) {
        int leader_lane = __ffs(active_mask) - 1;
        ChunkKey current_key = group.shfl(my_key, leader_lane);
        
        auto it = map_ref.find(group, current_key);
        

        if (my_key == current_key) {
            if (it != map_ref.end() && it->second != invalid_chunk_ptr() && it->second != nullptr) {
                uint32_t intra_chunk_idx = get_intra_chunk_index(global_x, global_y, global_z);
                d_output_block[output_idx] = it->second[intra_chunk_idx];
            } else {
                d_output_block[output_idx] = default_voxel_value();
            }
        }
        unsigned int processed_mask = group.ballot(my_key == current_key);
        active_mask &= ~processed_mask;
    }
}

std::vector<VoxelType> GpuHashMap::extract_grid_block(const int* aabb_indices) {
    int min_x = aabb_indices[0];
    int max_x = aabb_indices[1];
    int min_y = aabb_indices[2];
    int max_y = aabb_indices[3];
    int min_z = aabb_indices[4];
    int max_z = aabb_indices[5];

    size_t aabb_size_x = max_x - min_x + 1;
    size_t aabb_size_y = max_y - min_y + 1;
    size_t aabb_size_z = max_z - min_z + 1;
    
    size_t total_elements = aabb_size_x * aabb_size_y * aabb_size_z;
    
    if (total_elements == 0) {
        return {};
    }

    std::vector<VoxelType> h_block(total_elements);
    VoxelType* d_output_block;
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_block, total_elements * sizeof(VoxelType)));

    dim3 block_dim(32, 8, 1);
    dim3 grid_dim(
        (aabb_size_x + block_dim.x - 1) / block_dim.x,
        (aabb_size_y + block_dim.y - 1) / block_dim.y,
        aabb_size_z
    );

    auto map_ref = d_voxel_map_->ref(cuco::op::find);

    std::cout << "Extracting block with dimensions: "
              << aabb_size_x << " x " << aabb_size_y << " x " << aabb_size_z
              << " (total elements: " << total_elements << ")" << std::endl;
    std::cout << "aabb indices: "
              << "min_x: " << min_x << ", max_x: " << max_x
              << ", min_y: " << min_y << ", max_y: " << max_y
              << ", min_z: " << min_z << ", max_z: " << max_z
              << std::endl;

    extract_block_kernel<<<grid_dim, block_dim>>>(
        map_ref,
        d_output_block,
        min_x, min_y, min_z,
        aabb_size_x, aabb_size_y,
        aabb_size_z
    );
    
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(
        h_block.data(), 
        d_output_block, 
        total_elements * sizeof(VoxelType),
        cudaMemcpyDeviceToHost
    ));

    CHECK_CUDA_ERROR(cudaFree(d_output_block));
    
    return h_block;
}