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
#include "voxel-mapping/map_utils.cuh"

namespace voxel_mapping {

class VoxelMapping::VoxelMappingImpl {
public:
    float resolution_;
    int occupancy_threshold_;
    cudaStream_t stream_ = nullptr;
    std::unique_ptr<GpuHashMap> voxel_map_;
    std::unique_ptr<UpdateGenerator> update_generator_;
    std::unique_ptr<GridProcessor> grid_processor_;

    VoxelMappingImpl(const VoxelMappingParams& params)
        : resolution_(params.resolution), occupancy_threshold_(params.occupancy_threshold)
    {
        if (resolution_ <= 0) {
            throw std::invalid_argument("Resolution must be positive");
        }
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
        voxel_map_ = std::make_unique<GpuHashMap>(params.chunk_capacity, params.log_odds_occupied, params.log_odds_free, params.log_odds_min, params.log_odds_max, params.occupancy_threshold, params.free_threshold);
        spdlog::info("Voxel map initialized on GPU with initial capacity {}", params.chunk_capacity);
        update_generator_ = std::make_unique<UpdateGenerator>(resolution_, params.min_depth, params.max_depth);
        spdlog::info("Update generator initialized with resolution: {}, min_depth: {}, max_depth: {}", resolution_, params.min_depth, params.max_depth);
        grid_processor_ = std::make_unique<GridProcessor>(params.occupancy_threshold, params.free_threshold, params.edt_max_distance);
        spdlog::info("Grid processor initialized with occupancy threshold: {}, free threshold: {}, EDT max distance: {}", params.occupancy_threshold, params.free_threshold, params.edt_max_distance);
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

        ExtractBinaryOp extract_op = {d_output_buffer, aabb_size_x, aabb_size_y, occupancy_threshold_, max_dim_sq};

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

    void query_free_chunk_capacity() {
        uint32_t current_freelist_count;
        voxel_map_->get_freelist_counter(&current_freelist_count);
        size_t freelist_capacity = voxel_map_->get_freelist_capacity();
        uint32_t threshold = static_cast<uint32_t>(freelist_capacity * 0.95);

        if (current_freelist_count >= threshold) {
            int3 current_chunk_pos = update_generator_->get_current_chunk_position();
            
            spdlog::info("Freelist usage ({}) is above 95% threshold ({}). Clearing distant chunks.", 
                         current_freelist_count, threshold);

            voxel_map_->clear_chunks(current_chunk_pos);
        }
    }

};

VoxelMapping::VoxelMapping(const VoxelMappingParams& params)
    : pimpl_(std::make_unique<VoxelMappingImpl>(params)) {
}

VoxelMapping::~VoxelMapping() = default;

VoxelMapping::VoxelMapping(VoxelMapping&&) = default;
VoxelMapping& VoxelMapping::operator=(VoxelMapping&&) = default;

void VoxelMapping::integrate_depth(const float* depth_image, const float* transform) {
    pimpl_->integrate_depth(depth_image, transform);
}

std::vector<VoxelType> VoxelMapping::extract_grid_block(const AABB& aabb) {
    return pimpl_->extract_grid_data<ExtractionType::Block>(aabb, SliceZIndices{});
}

std::vector<VoxelType> VoxelMapping::extract_grid_slices(const AABB& aabb, const SliceZIndices& slice_indices) {
    return pimpl_->extract_grid_data<ExtractionType::Slice>(aabb, slice_indices);
}

std::vector<int> VoxelMapping::extract_edt_block(const AABB& aabb) {
    return pimpl_->extract_edt_data<ExtractionType::Block>(aabb, SliceZIndices{});
}

std::vector<int> VoxelMapping::extract_edt_slice(const AABB& aabb, const SliceZIndices& slice_indices) {
    return pimpl_->extract_edt_data<ExtractionType::Slice>(aabb, slice_indices);
}

void VoxelMapping::query_free_chunk_capacity() {
    pimpl_->query_free_chunk_capacity();
}

void VoxelMapping::set_camera_properties(float fx, float fy, float cx, float cy, uint32_t width, uint32_t height) {
    pimpl_->update_generator_->set_camera_properties(fx, fy, cx, cy, width, height);
}

AABB VoxelMapping::get_current_aabb() const {
    return pimpl_->update_generator_->get_aabb();
}

Frustum VoxelMapping::get_frustum() const {
    return pimpl_->update_generator_->get_frustum();
}

} // namespace voxel_mapping