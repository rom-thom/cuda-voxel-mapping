#ifndef MAP_UTILS_CUH
#define MAP_UTILS_CUH

#include "voxel-mapping/internal_types.cuh"

namespace voxel_mapping {

/**
 * @brief Defines the default initial value for a voxel.
 * This is used to represent an unknown state.
 */
inline constexpr __device__ VoxelType default_voxel_value() {
    return 0;
}

/**
 * @brief Returns an invalid chunk pointer.
 * This is used to indicate that a chunk pointer value in the hash map is not valid.
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
inline __host__ __device__ int floor_div(int a, int n) {
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
 * Applies a mask to handle negative indices with two's complement representation.
 * This supports indices in the range of -2097152 to 2097151, which fits within the 21 bits used for each coordinate.
 */
__device__ inline uint64_t pack_indices_to_key(int chunk_ix, int chunk_iy, int chunk_iz) {
    constexpr uint64_t MASK = 0x1FFFFF;

    return ((static_cast<uint64_t>(chunk_ix) & MASK) << 42) |
           ((static_cast<uint64_t>(chunk_iy) & MASK) << 21) |
            (static_cast<uint64_t>(chunk_iz) & MASK);
}

/**
 * @brief Unpacks a ChunkKey into its corresponding indices.
 * This is used to retrieve the original chunk coordinates from the packed key.
 * Handles negative indices by extending the sign bit.
 */
inline __host__ __device__ int3 unpack_key_to_indices(ChunkKey key) {
    constexpr uint64_t MASK = 0x1FFFFF;
    constexpr int BITS = 21;
    constexpr uint64_t SIGN_BIT = 1ULL << (BITS - 1); 

    int3 indices;

    int ix = (key >> 42) & MASK;
    if (ix & SIGN_BIT) {
        ix |= ~MASK;
    }
    indices.x = ix;

    int iy = (key >> 21) & MASK;
    if (iy & SIGN_BIT) {
        iy |= ~MASK;
    }
    indices.y = iy;

    int iz = key & MASK;
    if (iz & SIGN_BIT) {
        iz |= ~MASK;
    }
    indices.z = iz;
    
    return indices;
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

} // namespace voxel_mapping

#endif // MAP_UTILS_CUH