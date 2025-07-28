#ifndef GPU_HASH_MAP_CUH
#define GPU_HASH_MAP_CUH

#include <cuco/static_map.cuh>
#include <voxel-mapping/types.hpp>
#include "voxel-mapping/internal_types.cuh"
#include <memory>

namespace voxel_mapping {

class GpuHashMap {
    public:
        /**
         * @brief Constructs a cuco static map for voxel mapping using chunkidx as the key and ChunkPtr as the value.
         * This map is used to manage the preallocated chunks in the global memory pool, by distributing ChunkPtrs to Chunks on demand.
         * @param capacity The number of chunks to preallocate in the global memory pool. Hash map size is set to capacity / 0.90 to ensure a max load factor of 0.90.
         * @param log_odds_occupied Log-odds update value for occupied voxels.
         * @param log_odds_free Log-odds update value for free voxels.
         * @param log_odds_min Clamped minimum log-odds value for voxels.
         * @param log_odds_max Clamped maximum log-odds value for voxels.
         * @param occupancy_threshold Threshold for occupancy to consider a voxel occupied.
         * @param free_threshold Threshold for occupancy to consider a voxel free.
         */
        GpuHashMap(size_t capacity, VoxelType log_odds_occupied, VoxelType log_odds_free, VoxelType log_odds_min, VoxelType log_odds_max, VoxelType occupancy_threshold, VoxelType free_threshold);
        ~GpuHashMap();

        GpuHashMap(const GpuHashMap&) = delete;
        GpuHashMap& operator=(const GpuHashMap&) = delete;

        GpuHashMap(GpuHashMap&&) = default;
        GpuHashMap& operator=(GpuHashMap&&) = default;

        /**
         * @brief Updates the voxel map with the given AABB update by finding or inserting the corresponding chunkptrs in the hashmap.
         * @param aabb_update The AABB update containing the grid and its dimensions and the pointer to the AABB grid.
         * @param stream The CUDA stream to use for the operation.
         */
        void launch_map_update_kernel(
            AABBUpdate aabb_update,
            cudaStream_t stream);

        /**
         * @brief Queries the hash map for the voxels within the specified AABB and extracts them into a device memory block.
         * @param d_output_block Pointer to the output block where the extracted voxels will be stored.
         * @param aabb The AABB defining the region to extract.
         */
        void extract_block_from_map(VoxelType* d_output_block, const AABB& aabb);

        /**
         * @brief Clears chunks from the hash map that either have an invalid chunk pointer or is far away from the current chunk position.
         * Clears chunks equal to 10% of the total chunk capacity.
         * @param current_chunk_pos The current chunk position in 3D grid coordinates.
         */
        void clear_chunks(const int3& current_chunk_pos);

        /**
         * @brief Retrieves the counter for freelist allocations.
         * This counter indicates how many chunks are currently allocated and used in the hash map.
         * @param freelist_counter Pointer to a uint32_t variable where the counter will be stored.
         */
        void get_freelist_counter(uint32_t* freelist_counter);

        /**
         * @brief Retrieves the capacity of the freelist (number of chunks).
         * The freelist is used to manage chunk allocations and deallocations.
         * @return The capacity of the freelist.
         */
        size_t get_freelist_capacity() const {
            return freelist_capacity_;
        }

    private:
        /**
         * @brief Helper function to retrieve all chunks from the hash map.
         * This function retrieves all chunk keys and their corresponding pointers from the hash map.
         * @param h_keys Vector to store the chunk keys.
         * @param h_values Vector to store the chunk pointers.
         */
        void retrieve_all_chunks(std::vector<ChunkKey>& h_keys, std::vector<ChunkPtr>& h_values);

        /**
         * @brief Helper function to prioritize chunks for clearing based on distance from the current chunk position and pointer validity.
         * This function sorts the chunks by their squared distance to the current chunk position and returns the top N chunks,
         * prioritizing chunks that have invalid pointers.
         * @param h_keys Vector of chunk keys.
         * @param h_values Vector of chunk pointers.
         * @param current_chunk_pos The current chunk position in 3D grid coordinates.
         * @return A vector of ChunkInfo containing the prioritized chunks for clearing.
         */
        std::vector<ChunkInfo> prioritize_chunks_for_clearing(
            const std::vector<ChunkKey>& h_keys,
            const std::vector<ChunkPtr>& h_values,
            const int3& current_chunk_pos);

        /**
         * @brief Executes the chunk removal operation by erasing chunks from the hash map,
         * setting the chunk memory to the default value, and deallocating the chunk pointers by returning them to the freelist.
         * @param keys_to_erase Vector of chunk keys to erase.
         * @param ptrs_to_deallocate Vector of chunk pointers to deallocate.
         */
        void execute_chunk_removal(
            const std::vector<ChunkKey>& keys_to_erase,
            const std::vector<ChunkPtr>& ptrs_to_deallocate);

        std::unique_ptr<ChunkMap> d_voxel_map_;
        VoxelType* global_memory_pool_ = nullptr;
        ChunkPtr* freelist_ = nullptr;
        uint32_t* freelist_counter_ = nullptr;
        uint32_t freelist_capacity_ = 0;
        size_t map_capacity_ = 0;
};

} // namespace voxel_mapping

#endif // GPU_HASH_MAP_CUH