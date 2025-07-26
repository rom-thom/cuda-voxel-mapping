#ifndef VOXEL_MAPPING_HPP
#define VOXEL_MAPPING_HPP

#include <cstdint>
#include <vector>
#include <memory>
#include "voxel-mapping/types.hpp"

namespace voxel_mapping {

class VoxelMapping {
public:
    /**
     * @brief Constructs a VoxelMapping object that acts as the interface to mapping operations.
     * @param map_chunk_capacity The number of voxel chunks to allocate memory for.
     * @param resolution The resolution of the voxel grid.
     * @param min_depth The minimum depth value for insertions from the depth image.
     * @param max_depth The maximum depth value for insertions from the depth image.
     * @param log_odds_occupied Log-odds update value for occupied voxels.
     * @param log_odds_free Log-odds update value for free voxels.
     * @param log_odds_min Clamped minimum log-odds value for voxels.
     * @param log_odds_max Clamped maximum log-odds value for voxels.
     * @param occupancy_threshold Threshold for occupancy to consider a voxel occupied.
     * @param free_threshold Threshold for occupancy to consider a voxel free.
     */
    VoxelMapping(size_t map_chunk_capacity, float resolution, float min_depth, float max_depth, VoxelType log_odds_occupied, VoxelType log_odds_free, VoxelType log_odds_min, VoxelType log_odds_max, VoxelType occupancy_threshold, VoxelType free_threshold);
    
    ~VoxelMapping();

    VoxelMapping(VoxelMapping&&);
    VoxelMapping& operator=(VoxelMapping&&);
    VoxelMapping(const VoxelMapping&) = delete;
    VoxelMapping& operator=(const VoxelMapping&) = delete;


    /**
     * @brief Sets the necessary camera properties for the voxel mapping to begin processing depth images.
     * @param fx Focal length in the x direction.
     * @param fy Focal length in the y direction.
     * @param cx Optical center x-coordinate.
     * @param cy Optical center y-coordinate.
     * @param width Width of the image.
     * @param height Height of the image.
     */
    void set_camera_properties(float fx, float fy, float cx, float cy, uint32_t width, uint32_t height);

    /**
     * @brief Integrates a depth image into the voxel map using the provided transformation.
     * The sensor frame required the z-direction to be defined as the view/forward direction aligning with camera conventions.
     * @param depth_image Pointer to the host memory containing the depth image.
     * @param transform Pointer to the host memory containing the world-to-camera transformation matrix.
     */
    void integrate_depth(const float* depth_image,  const float* transform);

    /**
     * @brief Extracts a 3D block of voxels from the voxel map based on the provided AABB.
     * @param aabb The AABB defining the region to extract defined by its minimum corner index and size.
     * @return A vector containing the voxel data for the specified AABB.
     */
    std::vector<VoxelType> get_3d_block(const AABB& aabb);

    /**
     * @brief Returns the current AABB's minimum index in world coordinates and its size in grid coordinates.
     * @return AABB struct containing:
     * - min_corner_index: A vector containing the minimum index {x, y, z} of the AABB in world coordinates.
     * - aabb_size: A vector containing the size {x, y, z} of the AABB in grid coordinates.
     */
    AABB get_current_aabb();

    // void extract_slice(const Eigen::VectorXi& indices, std::vector<float>& slice);

    // void extract_dilated_slice(const Eigen::VectorXi& indices, std::vector<float>& slice, int radius);

    // void extract_esdf(const Eigen::VectorXi& indices, std::vector<float>& esdf);
    
private:
    class VoxelMappingImpl;

    std::unique_ptr<VoxelMappingImpl> pimpl_;
};

} // namespace voxel_mapping

#endif // VOXEL_MAPPING_HPP