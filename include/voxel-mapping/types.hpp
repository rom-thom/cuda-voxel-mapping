#ifndef TYPES_HPP
#define TYPES_HPP

#include <cstdint>

namespace voxel_mapping {

/// @brief Represents the log-odds value of a single voxel.
using VoxelType = int;

/// @brief A simple 3D integer vector.
struct Vec3i {
    int x, y, z;
};

/// @brief AABB (Axis-Aligned Bounding Box) structure.
/// @struct AABB
/// @param min_corner_index An int vector of length 3 containing the minimum index {x, y, z} of the AABB in world coordinates.
/// @param aabb_size An int vector of length 3 containing the size {x, y, z} of the AABB in grid coordinates.
struct AABB {
    Vec3i min_corner_index;
    Vec3i size;
};

/// @brief A simple 3D float vector.
struct Vec3f {
    float x, y, z;
};

/// @brief Four points representing a frustum plane.
struct FrustumPlane {
    Vec3f tl, tr, bl, br;
};

/// @brief Frustum structure representing a view frustum in 3D space.
/// @struct Frustum
/// @param near_plane The near plane of the frustum, defined by four points in world coordinates.
/// @param far_plane The far plane of the frustum, defined by four points in world coordinates.
struct Frustum {
    FrustumPlane near_plane;
    FrustumPlane far_plane;
};

} // namespace voxel_mapping

#endif // TYPES_HPP