#ifndef MAP_UTILS_CUH
#define MAP_UTILS_CUH

#include "voxel-mapping/internal_types.cuh"

/**
 * @brief Defines the default initial value for a voxel.
 * This is used to represent an unknown state.
 */
inline constexpr __device__ VoxelType default_voxel_value() {
    return 0;
}

/**
 * @brief Defines the value for flagged voxels.
 * This is used to indicate that a voxel has already been updated.
 */
inline constexpr __device__ VoxelFlag voxel_flag_value() {
    return 1u;
}

/**
 * @brief Returns an invalid chunk pointer.
 * This is used to indicate that a chunk pointer is not valid.
 */
inline __device__ __host__ ChunkPtr invalid_chunk_ptr() {
    return reinterpret_cast<ChunkPtr>(0xFFFFFFFFFFFFFFFF);
}

/**
 * @brief Returns the dimension of a chunk.
 * This is used to define the size of a chunk in the voxel grid.
 */
inline constexpr __device__ __host__ uint32_t chunk_dim() {
    return 32u;
}

/**
 * @brief Performs integer division that correctly handles negative numbers (rounds towards negative infinity).
 * Standard C++ integer division truncates towards zero, which is incorrect for this use case.
 */
__device__ inline int floor_div(int a, int n) {
    int r = a / n;
    if ((a % n) != 0 && (a * n) < 0) {
        r--;
    }
    return r;
}

/**
 * @brief Performs the modulo operation that correctly handles negative numbers (always returns a positive result).
 * The standard C++ '%' operator can return negative results, which is invalid for array indexing.
 */
__device__ inline int positive_modulo(int i, int n) {
    return (i % n + n) % n;
}

/**
 * @brief Packs three int indices into a single uint64_t key.
 * This is used to create a unique identifier for a chunk based on its coordinates.
 */
__device__ inline uint64_t pack_indices_to_key(int chunk_ix, int chunk_iy, int chunk_iz) {
    // Valid for chunk ids in ranging from -1,048,576 to +1,048,575
    return (static_cast<uint64_t>(chunk_ix) << 42) | (static_cast<uint64_t>(chunk_iy) << 21) | static_cast<uint64_t>(chunk_iz);
}

/**
 * @brief Calculates the unique key for the chunk that contains the given global coordinate.
 */
__device__ inline ChunkKey get_chunk_key(int global_x, int global_y, int global_z) {
    int chunk_ix = floor_div(global_x, chunk_dim());
    int chunk_iy = floor_div(global_y, chunk_dim());
    int chunk_iz = floor_div(global_z, chunk_dim());

    return pack_indices_to_key(chunk_ix, chunk_iy, chunk_iz);
}

/**
 * @brief Calculates the flattened 1D index for a voxel within its local chunk.
 * This is used to access voxel data in a 1D array representation of the chunk.
 */
__device__ inline uint32_t chunk_1d_index(int x, int y, int z) {
    return (z * chunk_dim() * chunk_dim()) + (y * chunk_dim()) + x;
}

/**
 * @brief Calculates the flattened 1D index for a voxel within its local chunk.
 * This is used to access voxel data in a 1D array representation of the chunk.
 */
__device__ inline uint32_t get_intra_chunk_index(int global_x, int global_y, int global_z) {
    uint32_t local_x = positive_modulo(global_x, chunk_dim());
    uint32_t local_y = positive_modulo(global_y, chunk_dim());
    uint32_t local_z = positive_modulo(global_z, chunk_dim());

    return chunk_1d_index(local_x, local_y, local_z);
}

/**
 * @brief Helper to get the 1D index for a voxel within a dense AABB block.
 * 
 */
__device__ inline uint32_t block_1d_index(int tx, int ty, int tz, int size_x, int size_y) {
    return (tz * size_y * size_x) + (ty * size_x) + tx;
}

#endif // MAP_UTILS_CUH