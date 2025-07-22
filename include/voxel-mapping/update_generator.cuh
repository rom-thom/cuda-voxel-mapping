#ifndef UPDATE_GENERATOR_CUH
#define UPDATE_GENERATOR_CUH

#include "voxel-mapping/internal_types.cuh"
#include <cstdint>

class UpdateGenerator {
public:
    UpdateGenerator(size_t update_list_capacity, float voxel_resolution, float min_depth, float max_depth);
    ~UpdateGenerator();

    UpdateGenerator(const UpdateGenerator&) = delete;
    UpdateGenerator& operator=(const UpdateGenerator&) = delete;

    UpdateGenerator(UpdateGenerator&&) = default;
    UpdateGenerator& operator=(UpdateGenerator&&) = default;

    uint32_t generate_updates(
        const float* h_depth_image, 
        const float* transform,
        cudaStream_t stream);

    void set_camera_properties(
        float fx, float fy, float cx, float cy,
        uint32_t width, uint32_t height);

    VoxelUpdate* get_update_list() const {
        return d_update_list_;
    }

    uint32_t* get_update_counter() const {
        return d_update_counter_;
    }


    void reset_update_state(cudaStream_t stream);

private:
    void launch_raycasting_kernel(cudaStream_t stream);

    std::unique_ptr<VoxelUpdateSet> d_voxel_update_set_;
    uint32_t update_list_capacity_;
    VoxelUpdate* d_update_list_ = nullptr;
    uint32_t* d_update_counter_ = nullptr;
    uint32_t* d_seen_voxel_counter_ = nullptr;

    float* d_transform_ = nullptr;
    float* d_depth_ = nullptr;
    size_t depth_buffer_size_ = 0;
    uint32_t image_width_;
    uint32_t image_height_;
};

#endif // UPDATE_GENERATOR_CUH