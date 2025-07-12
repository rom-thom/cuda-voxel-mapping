#ifndef VOXEL_MAPPING_HPP
#define VOXEL_MAPPING_HPP

#include <cstdint>
#include <Eigen/Dense>
#include <vector>
#include <memory>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

extern "C" void launch_process_depth_kernels(
    const float* d_depth, int width, int height,
    const float* d_transform, float* d_voxel_grid, char* d_aabb,
    int min_x, int max_x, int min_y, int max_y, int min_z, int max_z,
    cudaStream_t stream);

extern "C" void launch_extract_2d_slice_kernel(
    const float* d_voxel_grid, float* d_slice,
    int min_x, int max_x, int min_y, int max_y, int min_z, int max_z, cudaStream_t stream);


extern "C" void launch_extract_dilated_2d_slice_kernel(
    const float* d_voxel_grid, float* d_slice,
    int min_x, int max_x, int min_y, int max_y, int min_z, int max_z, int dilation_size,
    cudaStream_t stream);

extern "C" void launch_extract_binary_slice_kernel(
    const float* d_voxel_grid, float* d_slice,
    int min_x, int max_x, int min_y, int max_y, int min_z, int max_z,
    cudaStream_t stream);

extern "C" void launch_edt_kernels(
    float* d_binary_slice, float* d_edt,
    int width, int height,
    cudaStream_t stream);
    
extern "C" void launch_initialize_float_kernel(
    float* arr, float value, size_t n);
    
class VoxelMapping {
public:
    VoxelMapping(float resolution, uint size_x, uint size_y, uint size_z, float min_depth, float max_depth, float log_odds_occupied, float log_odds_free, float log_odds_min, float log_odds_max, float occupancy_threshold, float free_threshold);
    
    ~VoxelMapping();

    VoxelMapping(VoxelMapping&&);
    VoxelMapping& operator=(VoxelMapping&&);
    VoxelMapping(const VoxelMapping&) = delete;
    VoxelMapping& operator=(const VoxelMapping&) = delete;

    void set_K(float fx, float fy, float cx, float cy);
    void set_image_size(int width, int height);

    void integrate_depth(const float* depth_image, const Eigen::Matrix4f& transform, const Eigen::VectorXi& aabb_indices);
    
    std::vector<float> get_grid_block(const Eigen::VectorXi& aabb_indices);

    void extract_slice(const Eigen::VectorXi& indices, std::vector<float>& slice);

    void extract_dilated_slice(const Eigen::VectorXi& indices, std::vector<float>& slice, int radius);

    void extract_esdf(const Eigen::VectorXi& indices, std::vector<float>& esdf);
    
private:
    class VoxelMappingImpl;

    std::unique_ptr<VoxelMappingImpl> pimpl_;
};

#endif // VOXEL_MAPPING_HPP