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

} // namespace voxel_mapping

#endif // TYPES_HPP