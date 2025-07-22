#ifndef VOXEL_MAPPING_HPP
#define VOXEL_MAPPING_HPP

#include <cstdint>
#include <vector>
#include <memory>
#include "voxel-mapping/types.hpp"
    
class VoxelMapping {
public:
    VoxelMapping(size_t map_chunk_capacity, size_t voxel_update_capacity, float resolution, float min_depth, float max_depth, VoxelType log_odds_occupied, VoxelType log_odds_free, VoxelType log_odds_min, VoxelType log_odds_max, VoxelType occupancy_threshold, VoxelType free_threshold);
    
    ~VoxelMapping();

    VoxelMapping(VoxelMapping&&);
    VoxelMapping& operator=(VoxelMapping&&);
    VoxelMapping(const VoxelMapping&) = delete;
    VoxelMapping& operator=(const VoxelMapping&) = delete;

    void set_camera_properties(float fx, float fy, float cx, float cy, uint32_t width, uint32_t height);

    void integrate_depth(const float* depth_image,  const float* transform);

    std::vector<VoxelType> get_3d_block(const std::vector<VoxelType>& aabb);

    // void extract_slice(const Eigen::VectorXi& indices, std::vector<float>& slice);

    // void extract_dilated_slice(const Eigen::VectorXi& indices, std::vector<float>& slice, int radius);

    // void extract_esdf(const Eigen::VectorXi& indices, std::vector<float>& esdf);
    
private:
    class VoxelMappingImpl;

    std::unique_ptr<VoxelMappingImpl> pimpl_;
};

#endif // VOXEL_MAPPING_HPP