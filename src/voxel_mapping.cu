#include "voxel-mapping/voxel_mapping.hpp"
#include <cuda_runtime.h>
#include "voxel-mapping/host_macros.hpp"
#include "voxel-mapping/types.hpp"
#include "voxel-mapping/gpu_hash_map.cuh"
#include <vector>
#include <cfloat>
#include <iostream>
#include <spdlog/spdlog.h>
#include "voxel-mapping/update_generator.cuh"
#include "voxel-mapping/grid_processor.cuh"

namespace voxel_mapping {

class VoxelMapping::VoxelMappingImpl {
public:
    float resolution_;
    cudaStream_t stream_ = nullptr;
    std::unique_ptr<GpuHashMap> voxel_map_;
    std::unique_ptr<UpdateGenerator> update_generator_;
    std::unique_ptr<GridProcessor> grid_processor_;

    VoxelMappingImpl(size_t map_chunk_capacity, float resolution, float min_depth, float max_depth, VoxelType log_odds_occupied, VoxelType log_odds_free, VoxelType log_odds_min, VoxelType log_odds_max, VoxelType occupancy_threshold, VoxelType free_threshold)
        : resolution_(resolution)
    {
        if (resolution <= 0) {
            throw std::invalid_argument("Resolution must be positive");
        }
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
        voxel_map_ = std::make_unique<GpuHashMap>(map_chunk_capacity, log_odds_occupied, log_odds_free, log_odds_min, log_odds_max, occupancy_threshold, free_threshold);
        spdlog::info("Voxel map initialized on GPU with initial capacity {}", map_chunk_capacity);
        update_generator_ = std::make_unique<UpdateGenerator>(resolution_, min_depth, max_depth);
        spdlog::info("Update generator initialized with resolution: {}, min_depth: {}, max_depth: {}", resolution_, min_depth, max_depth);
        grid_processor_ = std::make_unique<GridProcessor>(occupancy_threshold, free_threshold);
        spdlog::info("Grid processor initialized with occupancy threshold: {}, free threshold: {}", occupancy_threshold, free_threshold);
    }

    ~VoxelMappingImpl() {
        if (stream_) cudaStreamDestroy(stream_);
    }

    void integrate_depth(const float* depth_image, const float* transform) {
        AABBUpdate aabb_update = update_generator_->generate_updates(
            depth_image, 
            transform,
            stream_
        );

        CHECK_CUDA_ERROR(cudaGetLastError()); 

        voxel_map_->launch_map_update_kernel(
            aabb_update,
            stream_
        );
        
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));
        CHECK_CUDA_ERROR(cudaGetLastError());
    }

    std::vector<VoxelType> extract_grid_block(const AABB& aabb) {
        int aabb_size_x = aabb.size.x;
        int aabb_size_y = aabb.size.y;
        int aabb_size_z = aabb.size.z;
        
        size_t total_elements = aabb_size_x * aabb_size_y * aabb_size_z;

        std::vector<VoxelType> h_block(total_elements);
        VoxelType* d_output_block;
        CHECK_CUDA_ERROR(cudaMalloc(&d_output_block, total_elements * sizeof(VoxelType)));

        voxel_map_->extract_block_from_map(d_output_block, aabb);

        CHECK_CUDA_ERROR(cudaMemcpy(
            h_block.data(), 
            d_output_block, 
            total_elements * sizeof(VoxelType),
            cudaMemcpyDeviceToHost
        ));

        CHECK_CUDA_ERROR(cudaFree(d_output_block));
        
        return h_block;
    }

    void extract_esdf_slice(const AABB& aabb_slice, std::vector<int>& esdf_slice) {
        int aabb_size_x = aabb_slice.size.x;
        int aabb_size_y = aabb_slice.size.y;
        int aabb_size_z = aabb_slice.size.z;

        size_t total_elements = aabb_size_x * aabb_size_y * aabb_size_z;
        size_t slice_size = aabb_size_x * aabb_size_y;

        VoxelType* d_aabb;
        CHECK_CUDA_ERROR(cudaMalloc(&d_aabb, total_elements * sizeof(VoxelType)));

        voxel_map_->extract_block_from_map(d_aabb, aabb_slice);

        int* d_binary_slice;
        CHECK_CUDA_ERROR(cudaMalloc(&d_binary_slice, slice_size * sizeof(int)));

        grid_processor_->launch_extract_binary_slice_kernel(
            d_aabb,
            d_binary_slice,
            aabb_slice.min_corner_index.x,
            aabb_slice.min_corner_index.y,
            aabb_slice.min_corner_index.z,
            aabb_size_x,
            aabb_size_y,
            aabb_size_z,
            stream_
        );

        int* d_esdf_slice;
        CHECK_CUDA_ERROR(cudaMalloc(&d_esdf_slice, slice_size * sizeof(int)));

        grid_processor_->launch_edt_kernels(
            d_binary_slice,
            d_esdf_slice,
            aabb_size_x,
            aabb_size_y,
            stream_
        );

        esdf_slice.resize(slice_size);
        CHECK_CUDA_ERROR(cudaMemcpy(
            esdf_slice.data(), 
            d_esdf_slice, 
            slice_size * sizeof(int),
            cudaMemcpyDeviceToHost
        ));
    }

};

VoxelMapping::VoxelMapping(size_t map_update_capacity, float resolution, float min_depth, float max_depth, VoxelType log_odds_occupied, VoxelType log_odds_free, VoxelType log_odds_min, VoxelType log_odds_max, VoxelType occupancy_threshold, VoxelType free_threshold)
: pimpl_(std::make_unique<VoxelMappingImpl>(map_update_capacity, resolution, min_depth, max_depth, log_odds_occupied, log_odds_free, log_odds_min, log_odds_max, occupancy_threshold, free_threshold))
{
}

VoxelMapping::~VoxelMapping() = default;

VoxelMapping::VoxelMapping(VoxelMapping&&) = default;
VoxelMapping& VoxelMapping::operator=(VoxelMapping&&) = default;

void VoxelMapping::integrate_depth(const float* depth_image, const float* transform) {
    pimpl_->integrate_depth(depth_image, transform);
}

void VoxelMapping::set_camera_properties(float fx, float fy, float cx, float cy, uint32_t width, uint32_t height) {
    pimpl_->update_generator_->set_camera_properties(fx, fy, cx, cy, width, height);
}

std::vector<VoxelType> VoxelMapping::get_3d_block(const AABB& aabb) {
    return pimpl_->extract_grid_block(aabb);
}

void VoxelMapping::extract_esdf_slice(const AABB& aabb, std::vector<int>& esdf_slice) {
    pimpl_->extract_esdf_slice(aabb, esdf_slice);
}

AABB VoxelMapping::get_current_aabb() const {
    AABB aabb;
    int3 min_corner = pimpl_->update_generator_->get_aabb_min_index();
    int3 size = pimpl_->update_generator_->get_aabb_size();
    Vec3i min_corner_index = {min_corner.x, min_corner.y, min_corner.z};
    Vec3i aabb_size = {size.x, size.y, size.z};
    aabb.min_corner_index = min_corner_index;
    aabb.size = aabb_size;
    return aabb;
}

Frustum VoxelMapping::get_frustum() const {
    return pimpl_->update_generator_->get_frustum();
}

} // namespace voxel_mapping