#ifndef GPU_HASH_MAP_CUH
#define GPU_HASH_MAP_CUH

#include <cuco/static_map.cuh>
#include <voxel-mapping/types.hpp>
#include "voxel-mapping/internal_types.cuh"
#include <memory>

namespace voxel_mapping {

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
            AABBUpdate aabb_update,
            cudaStream_t stream);

        void extract_block_from_map(VoxelType* d_output_block, const AABB& aabb);

    private:

        std::unique_ptr<ChunkMap> d_voxel_map_;
        VoxelType* global_memory_pool_ = nullptr;
        ChunkPtr* freelist_ = nullptr;
        uint32_t* freelist_counter_ = nullptr;
        uint32_t freelist_capacity_ = 0;
        size_t map_capacity_ = 0;
};

} // namespace voxel_mapping

#endif // GPU_HASH_MAP_CUH