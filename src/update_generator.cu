#include "voxel-mapping/update_generator.cuh"
#include "voxel-mapping/host_macros.hpp"
#include "voxel-mapping/map_utils.cuh"
#include "voxel-mapping/raycasting_utils.cuh"
#include "voxel-mapping/internal_types.cuh"
#include "cooperative_groups.h"
#include <limits.h>
#include <cassert>

namespace voxel_mapping {

static __constant__ float d_fx;
static __constant__ float d_fy;
static __constant__ float d_cx;
static __constant__ float d_cy;
static __constant__ uint32_t d_image_width;
static __constant__ uint32_t d_image_height;
static __constant__ float d_resolution;
static __constant__ float d_min_depth;
static __constant__ float d_max_depth;

UpdateGenerator::UpdateGenerator(float voxel_resolution, float min_depth, float max_depth) :
    voxel_resolution_(voxel_resolution),
    min_depth_(min_depth),
    max_depth_(max_depth)
{
    CHECK_CUDA_ERROR(cudaMalloc(&d_transform_, 16 * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_resolution, &voxel_resolution_, sizeof(float), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_min_depth, &min_depth_, sizeof(float), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_max_depth, &max_depth_, sizeof(float), 0, cudaMemcpyHostToDevice));
}

UpdateGenerator::~UpdateGenerator() {
    cudaFree(d_depth_);
    cudaFree(d_transform_);
    cudaFree(d_aabb_);
}

void UpdateGenerator::set_camera_properties(
    float fx, float fy, float cx, float cy,
    uint32_t width, uint32_t height) {
    fx_ = fx;
    fy_ = fy;
    cx_ = cx;
    cy_ = cy;
    image_width_ = width;
    image_height_ = height;

    cudaFree(d_depth_);
    cudaFree(d_aabb_);

    depth_buffer_size_ = static_cast<size_t>(image_width_) * image_height_ * sizeof(float);
    CHECK_CUDA_ERROR(cudaMalloc(&d_depth_, depth_buffer_size_));

    float frustum_width = (image_width_ / fx_) * max_depth_;
    float frustum_height = (image_height_ / fy_) * max_depth_;

    float space_diagonal = sqrtf(frustum_width*frustum_width + frustum_height*frustum_height + max_depth_*max_depth_);
    int aabb_max_dim_size = static_cast<int>(ceil(space_diagonal / voxel_resolution_)) + 1;
    
    aabb_max_size_ = {aabb_max_dim_size, aabb_max_dim_size, aabb_max_dim_size};
    uint32_t aabb_max_total_size = static_cast<uint32_t>(aabb_max_size_.x) * aabb_max_size_.y * aabb_max_size_.z;

    CHECK_CUDA_ERROR(cudaMalloc(&d_aabb_, aabb_max_total_size * sizeof(UpdateType)));

    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_fx, &fx_, sizeof(float), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_fy, &fy_, sizeof(float), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_cx, &cx_, sizeof(float), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_cy, &cy_, sizeof(float), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_image_width, &image_width_, sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_image_height, &image_height_, sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
}

void UpdateGenerator::compute_active_aabb(const float* transform) {
    FrustumBounds frustum_bounds = get_frustum_world_bounds(transform);
    set_aabb_origin_index_and_size(frustum_bounds);
}

FrustumBounds UpdateGenerator::get_frustum_world_bounds(const float* transform){
    float3 corners_cam[8];
    
    float near_x_min = (0.0f - cx_) * min_depth_ / fx_;
    float near_x_max = (image_width_ - cx_) * min_depth_ / fx_;
    float near_y_min = (0.0f - cy_) * min_depth_ / fy_;
    float near_y_max = (image_height_ - cy_) * min_depth_ / fy_;
    corners_cam[0] = {near_x_min, near_y_min, min_depth_};
    corners_cam[1] = {near_x_max, near_y_min, min_depth_};
    corners_cam[2] = {near_x_max, near_y_max, min_depth_};
    corners_cam[3] = {near_x_min, near_y_max, min_depth_};
    
    float far_x_min = (0.0f - cx_) * max_depth_ / fx_;
    float far_x_max = (image_width_ - cx_) * max_depth_ / fx_;
    float far_y_min = (0.0f - cy_) * max_depth_ / fy_;
    float far_y_max = (image_height_ - cy_) * max_depth_ / fy_;
    corners_cam[4] = {far_x_min, far_y_min, max_depth_};
    corners_cam[5] = {far_x_max, far_y_min, max_depth_};
    corners_cam[6] = {far_x_max, far_y_max, max_depth_};
    corners_cam[7] = {far_x_min, far_y_max, max_depth_};
    
    FrustumBounds bounds;
    bounds.min_frustum_corner = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
    bounds.max_frustum_corner = {std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()};
    bounds.max_near_plane_corner = {std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest()};
    
    for(int i = 0; i < 8; ++i) {
        float3 p_cam = corners_cam[i];
    
        float wx = transform[tf_index::basis_xx()] * p_cam.x + transform[tf_index::basis_yx()] * p_cam.y + transform[tf_index::basis_zx()] * p_cam.z + transform[tf_index::t_x()];
        float wy = transform[tf_index::basis_xy()] * p_cam.x + transform[tf_index::basis_yy()] * p_cam.y + transform[tf_index::basis_zy()] * p_cam.z + transform[tf_index::t_y()];
        float wz = transform[tf_index::basis_xz()] * p_cam.x + transform[tf_index::basis_yz()] * p_cam.y + transform[tf_index::basis_zz()] * p_cam.z + transform[tf_index::t_z()];
    
        bounds.min_frustum_corner.x = std::min(bounds.min_frustum_corner.x, wx);
        bounds.min_frustum_corner.y = std::min(bounds.min_frustum_corner.y, wy);
        bounds.min_frustum_corner.z = std::min(bounds.min_frustum_corner.z, wz);
        
        bounds.max_frustum_corner.x = std::max(bounds.max_frustum_corner.x, wx);
        bounds.max_frustum_corner.y = std::max(bounds.max_frustum_corner.y, wy);
        bounds.max_frustum_corner.z = std::max(bounds.max_frustum_corner.z, wz);
    
        if (i < 4) {
            bounds.max_near_plane_corner.x = std::max(bounds.max_near_plane_corner.x, wx);
            bounds.max_near_plane_corner.y = std::max(bounds.max_near_plane_corner.y, wy);
            bounds.max_near_plane_corner.z = std::max(bounds.max_near_plane_corner.z, wz);
        }
    }
    return bounds;
}

void UpdateGenerator::set_aabb_origin_index_and_size(FrustumBounds frustum_bounds) {
    int3 frustum_min_i = {
        static_cast<int>(std::floor(frustum_bounds.min_frustum_corner.x / voxel_resolution_)),
        static_cast<int>(std::floor(frustum_bounds.min_frustum_corner.y / voxel_resolution_)),
        static_cast<int>(std::floor(frustum_bounds.min_frustum_corner.z / voxel_resolution_))
    };

    int3 near_plane_max_i = {
        static_cast<int>(std::floor(frustum_bounds.max_near_plane_corner.x / voxel_resolution_)),
        static_cast<int>(std::floor(frustum_bounds.max_near_plane_corner.y / voxel_resolution_)),
        static_cast<int>(std::floor(frustum_bounds.max_near_plane_corner.z / voxel_resolution_))
    };

    int3 frustum_max_i = {
        static_cast<int>(std::floor(frustum_bounds.max_frustum_corner.x / voxel_resolution_)),
        static_cast<int>(std::floor(frustum_bounds.max_frustum_corner.y / voxel_resolution_)),
        static_cast<int>(std::floor(frustum_bounds.max_frustum_corner.z / voxel_resolution_))
    };

    int3 final_origin_i = frustum_min_i;

    int required_near_size_x = near_plane_max_i.x - frustum_min_i.x + 1;
    if (required_near_size_x > aabb_max_size_.x) {
        final_origin_i.x = near_plane_max_i.x - aabb_max_size_.x + 1;
    }

    int required_near_size_y = near_plane_max_i.y - frustum_min_i.y + 1;
    if (required_near_size_y > aabb_max_size_.y) {
        final_origin_i.y = near_plane_max_i.y - aabb_max_size_.y + 1;
    }

    int required_near_size_z = near_plane_max_i.z - frustum_min_i.z + 1;
    if (required_near_size_z > aabb_max_size_.z) {
        final_origin_i.z = near_plane_max_i.z - aabb_max_size_.z + 1;
    }
    
    aabb_min_index_ = final_origin_i;

    int required_full_size_x = frustum_max_i.x - final_origin_i.x + 1;
    aabb_current_size_.x = std::min(required_full_size_x, aabb_max_size_.x);

    int required_full_size_y = frustum_max_i.y - final_origin_i.y + 1;
    aabb_current_size_.y = std::min(required_full_size_y, aabb_max_size_.y);

    int required_full_size_z = frustum_max_i.z - final_origin_i.z + 1;
    aabb_current_size_.z = std::min(required_full_size_z, aabb_max_size_.z);
}

__global__ void mark_free_space_kernel(
    const float* d_depth, const float* d_transform, UpdateType* d_aabb_3d,
    int3 min_aabb_index, int3 aabb_current_size)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= d_image_width || y >= d_image_height) return;
    float depth = d_depth[image_1d_index(x, y, d_image_width)];
    if (depth < d_min_depth || depth <= 0.0f) return;
    if (depth > d_max_depth) depth = d_max_depth;

    float3 world_point = pixel_to_world_space(x, y, depth, d_transform, d_fx, d_fy, d_cx, d_cy);
    float3 start_point = {d_transform[tf_index::t_x()], d_transform[tf_index::t_y()], d_transform[tf_index::t_z()]};
    mark_ray_as_free(start_point, world_point, d_aabb_3d, min_aabb_index, aabb_current_size, d_resolution);
}

__global__ void mark_occupied_space_kernel(
    const float* d_depth, const float* d_transform, UpdateType* d_aabb_3d,
    int3 min_aabb_index, int3 aabb_current_size)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= d_image_width || y >= d_image_height) return;
    float depth = d_depth[image_1d_index(x, y, d_image_width)];
    if (depth < d_min_depth || depth <= 0.0f || depth > d_max_depth) return;

    float3 world_point = pixel_to_world_space(x, y, depth, d_transform, d_fx, d_fy, d_cx, d_cy);
    mark_endpoint_as_occupied(world_point, d_aabb_3d, min_aabb_index, aabb_current_size, d_resolution);
}

AABBUpdate UpdateGenerator::generate_updates(
    const float* h_depth_image, 
    const float* transform,
    cudaStream_t stream) 
{
    CHECK_CUDA_ERROR(cudaMemcpy(d_transform_, transform, 16 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_depth_, h_depth_image, depth_buffer_size_, cudaMemcpyHostToDevice));

    uint32_t aabb_max_total_size = static_cast<uint32_t>(aabb_max_size_.x) * aabb_max_size_.y * aabb_max_size_.z;
    CHECK_CUDA_ERROR(cudaMemset(d_aabb_, static_cast<int>(UpdateType::Unknown), aabb_max_total_size * sizeof(UpdateType)));

    compute_active_aabb(transform);

    dim3 threads(16, 16);
    dim3 blocks((image_width_ + threads.x - 1) / threads.x, (image_height_ + threads.y - 1) / threads.y);

    mark_free_space_kernel<<<blocks, threads, 0, stream>>>(
        d_depth_, d_transform_, d_aabb_, aabb_min_index_, aabb_current_size_);

    mark_occupied_space_kernel<<<blocks, threads, 0, stream>>>(
        d_depth_, d_transform_, d_aabb_, aabb_min_index_, aabb_current_size_);

    AABBUpdate aabb_update;
    aabb_update.d_aabb_grid = d_aabb_;
    aabb_update.aabb_min_index = aabb_min_index_;
    aabb_update.aabb_current_size = aabb_current_size_;
    return aabb_update;
}

} // namespace voxel_mapping