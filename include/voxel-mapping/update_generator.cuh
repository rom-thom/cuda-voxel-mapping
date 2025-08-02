#ifndef UPDATE_GENERATOR_CUH
#define UPDATE_GENERATOR_CUH

#include "voxel-mapping/internal_types.cuh"
#include <cstdint>

namespace voxel_mapping {

class UpdateGenerator {
public:

    /**
     * @brief Constructs an UpdateGenerator responsible for performing raycasting and generating updates for the AABB.
     * @param voxel_resolution The resolution of the voxels in meters.
     * @param min_depth The minimum depth value to consider for updates.
     * @param max_depth The maximum depth value to consider for updates.
     */
    UpdateGenerator(float voxel_resolution, float min_depth, float max_depth);
    ~UpdateGenerator();

    UpdateGenerator(const UpdateGenerator&) = delete;
    UpdateGenerator& operator=(const UpdateGenerator&) = delete;

    UpdateGenerator(UpdateGenerator&&) = default;
    UpdateGenerator& operator=(UpdateGenerator&&) = default;

    /**
     * @brief Orchestrates the generation of updates for the AABB grid based on the provided depth image and transformation.
     * @param h_depth_image Pointer to the host memory containing the depth image.
     * @param transform Pointer to the host memory containing the world-to-camera transformation matrix.
     * @param stream CUDA stream to use for asynchronous operations.
     */
    AABBUpdate generate_updates(
        const float* h_depth_image, 
        const float* transform,
        cudaStream_t stream);

    /**
     * @brief Sets the camera properties for the update generator and copies them to their respective device constants.
     * @param fx The focal length in the x direction.
     * @param fy The focal length in the y direction.
     * @param cx The optical center x-coordinate.
     * @param cy The optical center y-coordinate.
     * @param width The width of the image.
     * @param height The height of the image.
     */
    void set_camera_properties(
        float fx, float fy, float cx, float cy,
        uint32_t width, uint32_t height);

    /**
     * @brief Get the current AABB's minimum index in world coordinates.
     * @return A vector containing the minimum index {x, y, z} of the AABB.
     */
    int3 get_aabb_min_index() const {
        return aabb_min_index_;
    }

    /**
     * @brief Get the current AABB's size in grid coordinates.
     * @return A vector containing the size {x, y, z} of the AABB.
     */
    int3 get_aabb_size() const {
        return aabb_current_size_;
    }

    /**
     * @brief Get the defining properties of the AABB
     * @return A AABB object containing:
     * - min_corner_index: The minimum index {x, y, z} of the AABB in world coordinates.
     * - size: The size {x, y, z} of the AABB in grid coordinates.
     */
    AABB get_aabb() const {
        AABB aabb;
        aabb.min_corner_index = {aabb_min_index_.x, aabb_min_index_.y, aabb_min_index_.z};
        aabb.size = {aabb_current_size_.x, aabb_current_size_.y, aabb_current_size_.z};
        return aabb;
    }

    /**
     * @brief Get the frustum of the camera in world coordinates.
     * @return A Frustum object containing the near and far planes of the camera frustum.
     */
    Frustum get_frustum() const {
        return frustum_;
    }

    /**
     * @brief Get the current chunk position in world coordinates.
     * @return An int3 containing the current chunk position {x, y, z}.
     */
    int3 get_current_chunk_position() const {
        return current_chunk_pos_;
    }

private:
    /**
     * @brief Orchestrates the calculation of the current AABB min index and size
     *
     * This function calls the necessary helpers to compute the AABB's bounding
     * indices and stores the results in the class's member variables.
     * @param transform The current world-to-camera transformation matrix.
     */
    void compute_active_aabb(const float* transform);

    /**
     * @brief Calculates the bounding corners of the camera's view frustum in world space.
     * @param transform The current world-to-camera transformation matrix.
     * @return A FrustumBounds struct containing:
     * - min_frustum_corner: The minimum {x,y,z} corner of the entire frustum.
     * - max_frustum_corner: The maximum {x,y,z} corner of the entire frustum.
     * - max_near_plane_corner: The maximum {x,y,z} corner of only the near plane.
     */
    FrustumBounds get_frustum_world_bounds(const float* transform);

    /**
     * @brief Calculates the discrete grid index for the AABB's origin and its size
     *
     * The origin is determined from the frustum's bounds and may be shifted
     * to ensure the camera's near plane is fully contained within the AABB.
     * Sets the aabb_min_index_ and aabb_current_size_.
     * @param frustum_bounds The bounds of the frustum, containing:
     * - min_frustum_corner: The minimum {x,y,z} corner of the frustum.
     * - max_frustum_corner: The maximum {x,y,z} corner of the frustum.
     * - max_near_plane_corner: The maximum {x,y,z} corner of the near plane.
     */
    void set_aabb_origin_index_and_size(FrustumBounds frustum_bounds);

    /**
     * @brief Sets the current chunk position based on the provided transformation matrix.
     * The chunk position is calculated by flooring the translation components of the transformation
     * divided by the voxel resolution.
     * @param transform Pointer to the transformation matrix in host memory.
     */
    void set_current_chunk_position(const float* transform);

    float voxel_resolution_;
    float min_depth_;
    float max_depth_;

    UpdateType* d_aabb_ = nullptr;
    float* d_transform_ = nullptr;
    float* d_depth_ = nullptr;
    size_t depth_buffer_size_ = 0;
    uint32_t image_width_;
    uint32_t image_height_;
    float fx_;
    float fy_;
    float cx_;
    float cy_;
    uint32_t aabb_max_total_size_ = 0;
    int3 aabb_min_index_ = {0, 0, 0};
    int3 aabb_current_size_ = {0, 0, 0};
    int3 aabb_max_size_ = {0, 0, 0};
    Frustum frustum_;
    int3 current_chunk_pos_ = {0, 0, 0};
};

} // namespace voxel_mapping

#endif // UPDATE_GENERATOR_CUH