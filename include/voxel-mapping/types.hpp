#ifndef TYPES_HPP
#define TYPES_HPP

#include <cstdint>

namespace voxel_mapping {

/**
 * @brief Represents the log-odds value of a single voxel.
 */
using VoxelType = int;

/**
 * @brief A simple 3D integer vector.
 */
struct Vec3i {
    int x, y, z;
};

/**
 * @brief AABB (Axis-Aligned Bounding Box) structure defining the region to extract from the voxel map.
 */
struct AABB {
    ///< An int vector of length 3 containing the minimum index {x, y, z} of the AABB in world coordinates.
    Vec3i min_corner_index;
    ///< An int vector of length 3 containing the size {x, y, z} of the AABB in grid coordinates.
    Vec3i size;
};

/**
 * @brief A simple 3D float vector.
 */
struct Vec3f {
    float x, y, z;
};

/**
 * @brief Four points representing a frustum plane.
 */
struct FrustumPlane {
    Vec3f tl, tr, bl, br;
};

/**
 * @brief Frustum structure representing a view frustum in 3D space.
 */
struct Frustum {
    ///< The near plane of the frustum, defined by four points in world coordinates.
    FrustumPlane near_plane;
    ///< The far plane of the frustum, defined by four points in world coordinates.
    FrustumPlane far_plane;
};

/**
 * @brief Struct to define the z indices for a multi-slice extraction.
 * Allows for a maximum of 5 z indices to be extracted at once.
 * This is useful for extracting slices of the voxel map at different z levels.
 */
struct SliceZIndices {
    ///< Maximum number of z indices that can be stored.
    static constexpr int MAX_Z_INDICES = 5;
    ///< Array to hold the z indices.
    int indices[MAX_Z_INDICES];
    ///< Number of valid indices currently stored in the array.
    int count;
};

} // namespace voxel_mapping

#endif // TYPES_HPP