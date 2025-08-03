#ifndef VOXEL_MAPPING_IMPL_HPP
#define VOXEL_MAPPING_IMPL_HPP

#include "voxel-mapping/voxel_mapping.hpp"
#include <cuda_runtime.h>
#include <memory>

#include "voxel-mapping/gpu_hash_map.cuh"
#include "voxel-mapping/update_generator.cuh"
#include "voxel-mapping/grid_processor.cuh"
#include "voxel-mapping/types.hpp"
#include "voxel-mapping/internal_types.cuh"
#include "voxel-mapping/host_macros.hpp"
#include "voxel-mapping/map_utils.cuh"

namespace voxel_mapping {

class VoxelMappingImpl {
public:
    VoxelMappingImpl(const VoxelMappingParams& params);
    ~VoxelMappingImpl();

    void integrate_depth(const float* depth_image, const float* transform);

    void query_free_chunk_capacity();

    void set_camera_properties(float fx, float fy, float cx, float cy, uint32_t width, uint32_t height);
    AABB get_current_aabb() const;
    Frustum get_frustum() const;

    template <ExtractionType Type>
    std::vector<VoxelType> extract_grid_data(const AABB& aabb, const SliceZIndices& slice_indices) {
        const int aabb_size_x = aabb.size.x;
        const int aabb_size_y = aabb.size.y;
    
    int num_z_layers;
    if constexpr (Type == ExtractionType::Block) {
        num_z_layers = aabb.size.z;
    } else {
        num_z_layers = slice_indices.count;
    }

    const size_t total_elements = static_cast<size_t>(aabb_size_x) * aabb_size_y * num_z_layers;

    if (total_elements == 0) {
        return {};
    }

    std::vector<VoxelType> h_output(total_elements);
    VoxelType* d_output_buffer;
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_buffer, total_elements * sizeof(VoxelType)));

    ExtractOp extract_op = {d_output_buffer, aabb_size_x, aabb_size_y};

    voxel_map_->launch_map_extraction_kernel<Type>(
        aabb, 
        slice_indices, 
        extract_op
    );

    CHECK_CUDA_ERROR(cudaMemcpy(
        h_output.data(), 
        d_output_buffer, 
        total_elements * sizeof(VoxelType),
        cudaMemcpyDeviceToHost
    ));

    CHECK_CUDA_ERROR(cudaFree(d_output_buffer));
    
    return h_output;
}

template <ExtractionType Type>
std::vector<int> extract_edt_data(const AABB& aabb, const SliceZIndices& slice_indices) {
    const int aabb_size_x = aabb.size.x;
    const int aabb_size_y = aabb.size.y;
    
    int num_z_layers;
    int max_dim_sq = 0;
    if constexpr (Type == ExtractionType::Block) {
        num_z_layers = aabb.size.z;
        max_dim_sq = std::max({aabb_size_x, aabb_size_y, aabb.size.z});
        max_dim_sq *= max_dim_sq;
    } else {
        num_z_layers = slice_indices.count;
        max_dim_sq = std::max(aabb_size_x, aabb_size_y);
        max_dim_sq *= max_dim_sq;
    }

    const size_t total_elements = static_cast<size_t>(aabb_size_x) * aabb_size_y * num_z_layers;

    if (total_elements == 0) {
        return {};
    }

    std::vector<int> h_output(total_elements);
    int* d_output_buffer;
    CHECK_CUDA_ERROR(cudaMalloc(&d_output_buffer, total_elements * sizeof(int)));

    ExtractBinaryOp extract_op = {d_output_buffer, aabb_size_x, aabb_size_y, free_threshold_, max_dim_sq};

    voxel_map_->launch_map_extraction_kernel<Type>(aabb, slice_indices, extract_op);

    if constexpr (Type == ExtractionType::Block) {
        grid_processor_->launch_3d_edt_kernels(
            d_output_buffer, aabb_size_x, aabb_size_y, num_z_layers, stream_
        );
    } else {
        grid_processor_->launch_edt_slice_kernels(
            d_output_buffer, aabb_size_x, aabb_size_y, num_z_layers, stream_
        );
    }

    CHECK_CUDA_ERROR(cudaMemcpy(
        h_output.data(), 
        d_output_buffer, 
        total_elements * sizeof(int),
        cudaMemcpyDeviceToHost
    ));

    CHECK_CUDA_ERROR(cudaFree(d_output_buffer));
    
    return h_output;
}

private:
    float resolution_;
    int occupancy_threshold_;
    int free_threshold_;
    mutable std::shared_mutex gpu_mutex_;
    cudaStream_t stream_ = nullptr;
    std::unique_ptr<GpuHashMap> voxel_map_;
    std::unique_ptr<UpdateGenerator> update_generator_;
    std::unique_ptr<GridProcessor> grid_processor_;
};

}
#endif