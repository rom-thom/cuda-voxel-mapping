#ifndef INTERNAL_TYPES_CUH
#define INTERNAL_TYPES_CUH

#include "voxel-mapping/types.hpp"
#include <cstdint>
#include <cuco/static_map.cuh>

namespace voxel_mapping {

/// @brief Unique identifier for a voxel chunk, represented as a 64-bit unsigned integer.
using ChunkKey = uint64_t;


/// @brief Pointer to the first voxel in a chunk, represented as a pointer to VoxelType.
using ChunkPtr = VoxelType*;

/// @brief Defines the type of update to be applied to a voxel.
enum class UpdateType : uint8_t {
    Unknown,  ///< The voxel state has not been determined, do nothing.
    Free,     ///< The voxel is part of free space (ray passed through it).
    Occupied, ///< The voxel is occupied by a surface.
};

/// @brief Structure representing an update to an axis-aligned bounding box (AABB) region.
/// @struct AABBUpdate
/// @param d_aabb_grid Device pointer to the grid of update types for the AABB.
/// @param aabb_min_index Minimum index (corner) of the AABB in 3D grid coordinates.
/// @param aabb_current_size Current size of the AABB in grid units.
struct AABBUpdate {
    UpdateType* d_aabb_grid;
    int3 aabb_min_index;
    int3 aabb_current_size; 
};

/// @brief Structure representing the bounds of a camera frustum in 3D space.
/// @struct FrustumBounds
/// @param min_frustum_corner Minimum corner of the frustum in 3D space.
/// @param max_frustum_corner Maximum corner of the frustum in 3D space.
/// @param max_near_plane_corner Maximum corner of the near plane of the frustum in 3D space.
struct FrustumBounds {
    float3 min_frustum_corner;
    float3 max_frustum_corner;
    float3 max_near_plane_corner;
};

/// @brief Type alias for a hashmap with {key, value} pairs are {ChunkKey, ChunkPtr}.
/// This map is used to manage voxel chunks allowing for on-demand allocation of chunks.
using ChunkMap = cuco::static_map<
    ChunkKey,
    ChunkPtr,
    cuco::extent<std::size_t>,
    cuda::thread_scope_device,
    cuda::std::equal_to<ChunkKey>,
    cuco::linear_probing<32, cuco::default_hash_function<ChunkKey>>
>;

/// @brief Type alias for a reference to a ChunkMap.
/// This reference allows for read and write operations on the ChunkMap.
using ChunkMapRef = decltype(std::declval<ChunkMap&>().ref(cuco::op::find, cuco::op::insert_and_find));

/// @brief Type alias for a constant reference to a ChunkMap.
/// This reference allows for read-only operations on the ChunkMap.
using ConstChunkMapRef = decltype(std::declval<ChunkMap&>().ref(cuco::op::find));

} // namespace voxel_mapping

#endif // INTERNAL_TYPES_CUH