#include "voxel-mapping/gpu_hash_map.cuh"
#include "voxel-mapping/host_macros.hpp"
#include "voxel-mapping/map_utils.cuh"
#include "voxel-mapping/internal_types.cuh"
#include <algorithm>
#include <vector>
#include <cstdint>
#include <limits>

namespace voxel_mapping {

static __constant__ uint32_t d_freelist_capacity;
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

    freelist_capacity_ = capacity;

    map_capacity_ = static_cast<size_t>(static_cast<double>(freelist_capacity_) / 0.9);

    ChunkKey empty_key_sentinel = std::numeric_limits<ChunkKey>::max();
    ChunkPtr empty_value_sentinel = nullptr;

    d_voxel_map_ = std::make_unique<ChunkMap>(
        map_capacity_,
        cuco::empty_key{empty_key_sentinel},
        cuco::empty_value{empty_value_sentinel}
    );

    size_t chunk_size_elements = chunk_dim() * chunk_dim() * chunk_dim();
    size_t chunk_size_bytes = sizeof(VoxelType) * chunk_size_elements;
    size_t pool_size_bytes = chunk_size_bytes * freelist_capacity_;

    CHECK_CUDA_ERROR(cudaMalloc(&global_memory_pool_, pool_size_bytes));
    CHECK_CUDA_ERROR(cudaMalloc(&freelist_, sizeof(ChunkPtr) * freelist_capacity_));
    CHECK_CUDA_ERROR(cudaMalloc(&freelist_counter_, sizeof(uint32_t)));

    CHECK_CUDA_ERROR(cudaMemset(global_memory_pool_, default_voxel_value(), pool_size_bytes));

    std::vector<ChunkPtr> h_chunk_ptrs(freelist_capacity_);
    for (size_t i = 0; i < freelist_capacity_; ++i) {
        h_chunk_ptrs[i] = global_memory_pool_ + (i * chunk_size_elements);
    }

    CHECK_CUDA_ERROR(cudaMemcpy(
        freelist_,
        h_chunk_ptrs.data(),
        sizeof(ChunkPtr) * freelist_capacity_,
        cudaMemcpyHostToDevice
    ));

    uint32_t initial_count = 0;
    CHECK_CUDA_ERROR(cudaMemcpy(
        freelist_counter_,
        &initial_count,
        sizeof(uint32_t),
        cudaMemcpyHostToDevice
    ));

    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_freelist_capacity, &freelist_capacity_, sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_log_odds_occupied, &log_odds_occupied, sizeof(VoxelType), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_log_odds_free, &log_odds_free, sizeof(VoxelType), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_log_odds_min, &log_odds_min, sizeof(VoxelType), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_log_odds_max, &log_odds_max, sizeof(VoxelType), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_occupancy_threshold, &occupancy_threshold, sizeof(VoxelType), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_free_threshold, &free_threshold, sizeof(VoxelType), 0, cudaMemcpyHostToDevice));
}

GpuHashMap::~GpuHashMap() = default;

__device__ inline ChunkPtr allocate_chunk_from_pool(
    ChunkPtr* freelist, uint32_t* counter, uint32_t freelist_capacity) {
    uint32_t index = *((volatile uint32_t*)counter);

    while (index < freelist_capacity) {
        uint32_t old_value = atomicCAS(counter, index, index + 1);
        if (old_value == index) {
            return freelist[index];
        }
        index = old_value;
    }
    return nullptr;
}

__device__ inline void deallocate_chunk_to_pool(ChunkPtr chunk_ptr, ChunkPtr* freelist, uint32_t* counter) {
    uint32_t index = atomicSub(counter, 1);
    freelist[index - 1] = chunk_ptr;
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
    if (update_type == UpdateType::Unknown) return;
    VoxelType update_value = (update_type == UpdateType::Free) ? d_log_odds_free : d_log_odds_occupied;
    atomic_add_and_clamp(
        &chunk_ptr[intra_chunk_idx],
        update_value,
        d_log_odds_min,
        d_log_odds_max
    );
}

__global__ void update_map_from_aabb(
    ChunkMapRef map_ref,
    ChunkPtr* freelist,
    uint32_t* freelist_counter,
    UpdateType* d_aabb,
    int3 aabb_min_index,
    int3 aabb_size)
{
    auto group = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());

    int aabb_x = blockIdx.x * blockDim.x + threadIdx.x;
    int aabb_y = blockIdx.y * blockDim.y + threadIdx.y;
    int aabb_z = blockIdx.z;

    bool is_in_bounds = (aabb_x < aabb_size.x && aabb_y < aabb_size.y && aabb_z < aabb_size.z);

    ChunkKey my_key = 0;
    uint32_t intra_chunk_idx = 0;
    UpdateType update_type = UpdateType::Unknown;

    if (is_in_bounds) {
        int global_x = aabb_min_index.x + aabb_x;
        int global_y = aabb_min_index.y + aabb_y;
        int global_z = aabb_min_index.z + aabb_z;

        my_key = get_chunk_key(global_x, global_y, global_z);
        intra_chunk_idx = get_intra_chunk_index(global_x, global_y, global_z);
        update_type = d_aabb[block_1d_index(aabb_x, aabb_y, aabb_z, aabb_size.x, aabb_size.y)];
    }

    unsigned int active_mask = group.ballot(is_in_bounds && update_type != UpdateType::Unknown);
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
                    ChunkPtr new_chunk_ptr = allocate_chunk_from_pool(freelist, freelist_counter, d_freelist_capacity);

                    if (new_chunk_ptr != nullptr) {
                        auto ref = cuda::atomic_ref<ChunkPtr, cuda::thread_scope_device>{insert_it->second};
                        ref.exchange(new_chunk_ptr, cuda::memory_order_release);
                        final_chunk_ptr = new_chunk_ptr;
                    } else {
                        // TODO: Handle allocation failure
                    }
                
                }
            final_chunk_ptr = group.shfl(final_chunk_ptr, leader_lane);
            } else {
                final_chunk_ptr = insert_it->second;
            }
        }

        if (my_key == current_key && final_chunk_ptr != invalid_chunk_ptr() && final_chunk_ptr != nullptr) {
            update_voxels(final_chunk_ptr, intra_chunk_idx, update_type);
        }

        unsigned int processed_mask = group.ballot(my_key == current_key);
        active_mask &= ~processed_mask;
    }
}

void GpuHashMap::launch_map_update_kernel(
    AABBUpdate aabb_update,
    cudaStream_t stream)
{
    auto map_ref = d_voxel_map_->ref(cuco::op::find, cuco::op::insert_and_find);

    int3 aabb_size = {
        aabb_update.aabb_current_size.x,
        aabb_update.aabb_current_size.y,
        aabb_update.aabb_current_size.z
    };

    dim3 block_dim(32, 8, 1);
    dim3 grid_dim(
        (aabb_size.x + block_dim.x - 1) / block_dim.x,
        (aabb_size.y + block_dim.y - 1) / block_dim.y,
        aabb_size.z
    );

    update_map_from_aabb<<<grid_dim, block_dim, 0, stream>>>(
        map_ref,
        freelist_,
        freelist_counter_,
        aabb_update.d_aabb_grid,
        aabb_update.aabb_min_index,
        aabb_size);
    CHECK_CUDA_ERROR(cudaGetLastError());
}

__global__ void extract_block_kernel(
    ConstChunkMapRef map_ref,
    VoxelType* d_output_block,
    int min_x, int min_y, int min_z,
    int size_x, int size_y, int size_z)
{
    auto group = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());

    int aabb_x = blockIdx.x * blockDim.x + threadIdx.x;
    int aabb_y = blockIdx.y * blockDim.y + threadIdx.y;
    int aabb_z = blockIdx.z;

    bool is_in_bounds = (aabb_x < size_x && aabb_y < size_y && aabb_z < size_z);

    ChunkKey my_key = 0;
    uint32_t output_idx = 0;
    
    if (is_in_bounds) {
        int global_x = min_x + aabb_x;
        int global_y = min_y + aabb_y;
        int global_z = min_z + aabb_z;

        my_key = get_chunk_key(global_x, global_y, global_z);
        output_idx = block_1d_index(aabb_x, aabb_y, aabb_z, size_x, size_y);
    }
    
    unsigned int active_mask = group.ballot(is_in_bounds);
    
    while(active_mask != 0) {
        int leader_lane = __ffs(active_mask) - 1;
        ChunkKey current_key = group.shfl(my_key, leader_lane);
        
        auto it = map_ref.find(group, current_key);
        

        if (my_key == current_key) {
            if (it != map_ref.end() && it->second != invalid_chunk_ptr() && it->second != nullptr) {
                int global_x = min_x + aabb_x;
                int global_y = min_y + aabb_y;
                int global_z = min_z + aabb_z;
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
void GpuHashMap::extract_block_from_map(VoxelType* d_output_block, const AABB& aabb) {

    int min_x = aabb.min_corner_index.x;
    int min_y = aabb.min_corner_index.y;
    int min_z = aabb.min_corner_index.z;
    int aabb_size_x = aabb.size.x;
    int aabb_size_y = aabb.size.y;
    int aabb_size_z = aabb.size.z;

    dim3 block_dim(32, 8, 1);
    dim3 grid_dim(
        (aabb_size_x + block_dim.x - 1) / block_dim.x,
        (aabb_size_y + block_dim.y - 1) / block_dim.y,
        (aabb_size_z + block_dim.z - 1) / block_dim.z
    );

    auto map_ref = d_voxel_map_->ref(cuco::op::find);

    extract_block_kernel<<<grid_dim, block_dim>>>(
        map_ref,
        d_output_block,
        min_x, min_y, min_z,
        aabb_size_x, aabb_size_y,
        aabb_size_z
    );
}

} // namespace voxel_mapping