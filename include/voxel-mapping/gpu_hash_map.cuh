#ifndef GPU_HASH_MAP_CUH
#define GPU_HASH_MAP_CUH

#include <cuco/static_map.cuh>
#include "voxel-mapping/internal_types.cuh"
#include <memory>

class GpuHashMap {
    public:
        GpuHashMap(size_t capacity, VoxelType log_odds_occupied, VoxelType log_odds_free, VoxelType log_odds_min, VoxelType log_odds_max, VoxelType occupancy_threshold, VoxelType free_threshold);
        ~GpuHashMap();

        GpuHashMap(const GpuHashMap&) = delete;
        GpuHashMap& operator=(const GpuHashMap&) = delete;

        GpuHashMap(GpuHashMap&&) = default;
        GpuHashMap& operator=(GpuHashMap&&) = default;
        
        ChunkMap* get_map();

        void launch_map_update_kernel(
            uint32_t num_updates,
            const VoxelUpdate* update_list,
            cudaStream_t stream);

        std::vector<VoxelType> extract_grid_block(const VoxelType* aabb_indices);

    private:

        std::unique_ptr<ChunkMap> d_voxel_map_;
        size_t map_capacity_;
        VoxelType* global_memory_pool_ = nullptr;
        ChunkPtr* freelist_ = nullptr;
        uint32_t* freelist_counter_ = nullptr;
        uint32_t* freelist_allocation_counter_ = nullptr;
        uint32_t* freelist_capacity_ = nullptr;

};

#endif // GPU_HASH_MAP_CUH