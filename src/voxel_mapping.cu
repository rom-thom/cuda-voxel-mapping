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

class VoxelMapping::VoxelMappingImpl {
public:
    float resolution_;
    cudaStream_t stream_ = nullptr;
    std::unique_ptr<GpuHashMap> voxel_map_;
    std::unique_ptr<UpdateGenerator> update_generator_;

    VoxelMappingImpl(size_t map_chunk_capacity, size_t voxel_update_capacity, float resolution, float min_depth, float max_depth, VoxelType log_odds_occupied, VoxelType log_odds_free, VoxelType log_odds_min, VoxelType log_odds_max, VoxelType occupancy_threshold, VoxelType free_threshold)
        : resolution_(resolution)
    {
        if (resolution <= 0) {
            throw std::invalid_argument("Resolution must be positive");
        }
        CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));
        voxel_map_ = std::make_unique<GpuHashMap>(map_chunk_capacity, log_odds_occupied, log_odds_free, log_odds_min, log_odds_max, occupancy_threshold, free_threshold);
        spdlog::info("Voxel map initialized on GPU with initial capacity {}", map_chunk_capacity);
        update_generator_ = std::make_unique<UpdateGenerator>(voxel_update_capacity, resolution_, min_depth, max_depth);
        spdlog::info("Update generator initialized on GPU with capacity {}", voxel_update_capacity);
    }

    ~VoxelMappingImpl() {
        if (stream_) cudaStreamDestroy(stream_);
    }

    void integrate_depth(const float* depth_image, const float* transform) {
        std::cout << "Starting generating updates from depth image." << std::endl;
        uint32_t num_updates = update_generator_->generate_updates(
            depth_image, 
            transform,
            stream_
        );

        CHECK_CUDA_ERROR(cudaGetLastError());

        std::cout << "Generated " << num_updates << " updates from depth image." << std::endl;

        if (num_updates == 0) {
            spdlog::warn("No updates generated from depth image.");
            return; 
        }

        const VoxelUpdate* d_update_list = update_generator_->get_update_list();

        voxel_map_->launch_map_update_kernel(
            num_updates,
            d_update_list,
            stream_
        );
        
        CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));
        spdlog::info("Launched map update kernel with {} updates.", num_updates);
    }

};

    // std::vector<float> get_grid_block(const Eigen::VectorXi& aabb_indices) {
    //     CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));
        
    //     int grid_min_x = aabb_indices[0];
    //     int grid_max_x = aabb_indices[1];
    //     int grid_min_y = aabb_indices[2];
    //     int grid_max_y = aabb_indices[3];
    //     int grid_min_z = aabb_indices[4];
    //     int grid_max_z = aabb_indices[5];
        
    //     size_t aabb_size_x = grid_max_x - grid_min_x + 1;
    //     size_t aabb_size_y = grid_max_y - grid_min_y + 1;
    //     size_t aabb_size_z = grid_max_z - grid_min_z + 1;
        
    //     size_t total_elements = aabb_size_x * aabb_size_y * aabb_size_z;
    //     std::vector<float> block(total_elements);
        
    //     cudaMemcpy3DParms copyParams = {0};
        
    //     // Source: d_voxel_grid (Z-X-Y major order: Z varies fastest, then X, then Y)
    //     // The pitch is the stride between X-planes (size_z * sizeof(float))
    //     copyParams.srcPtr = make_cudaPitchedPtr(
    //         d_voxel_grid_ + (grid_min_y * size_x_ * size_z_ + grid_min_x * size_z_ + grid_min_z),
    //         size_z_ * sizeof(float), // Pitch (stride in bytes between x-planes)
    //         size_z_,                 // Width in elements (y-dimension)
    //         size_x_                  // Height in elements (x-dimension)
    //     );
        
    //     // Destination: block (Z-X-Y major order)
    //     // Since block is a flat vector, the pitch matches the z-dimension size
    //     copyParams.dstPtr = make_cudaPitchedPtr(
    //         block.data(),
    //         aabb_size_z * sizeof(float), // Pitch (stride in bytes between x-planes in AABB)
    //         aabb_size_z,                 // Width in elements (y-dimension of AABB)
    //         aabb_size_x                  // Height in elements (x-dimension of AABB)
    //     );
        
    //     // Extent of the region to copy (AABB subregion in Z-X-Y major order)
    //     copyParams.extent = make_cudaExtent(
    //         aabb_size_z * sizeof(float), // Width in bytes (z-dimension)
    //         aabb_size_x,                 // Height in elements (x-dimension)
    //         aabb_size_y                  // Depth in elements (y-dimension)
    //     );
        
    //     copyParams.kind = cudaMemcpyDeviceToHost;
        
    //     CHECK_CUDA_ERROR(cudaMemcpy3D(&copyParams));
        
    //     return block;
    // }

    
        // launch_process_depth_kernels(d_depth, image_width_, image_height_,
        //     d_transform, d_voxel_grid_, d_aabb_,
        //     aabb_indices[0], aabb_indices[1], aabb_indices[2], aabb_indices[3], aabb_indices[4], aabb_indices[5],
        //     stream_);
    
        // Synchronize before freeing memory
    //     CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));
    //     cudaFree(d_buffer_);
    //     d_buffer_ = nullptr; // Reset pointer to avoid dangling references
    //     cudaFree(d_aabb_);
    //     d_aabb_ = nullptr; // Reset pointer to avoid dangling references
    // }
    
    // void extract_slice(const Eigen::VectorXi& indices, std::vector<float>& slice) {
    //     int min_x = indices[0];
    //     int max_x = indices[1];
    //     int min_y = indices[2];
    //     int max_y = indices[3];
    //     int min_z = indices[4];
    //     int max_z = indices[5];
        
    //     size_t slice_size_x = max_x - min_x + 1;
    //     size_t slice_size_y = max_y - min_y + 1;
        
    //     size_t slice_size = slice_size_x * slice_size_y;
        
    //     slice.resize(slice_size);
        
    //     float* d_slice;
        
    //     CHECK_CUDA_ERROR(cudaMalloc(&d_slice, slice_size * sizeof(float)));
    //     CHECK_CUDA_ERROR(cudaMemset(d_slice, -1.0, slice_size * sizeof(float)));
        
    //     launch_extract_2d_slice_kernel(d_voxel_grid_, d_slice, min_x, max_x, min_y, max_y, min_z, max_z, stream_);
        
    //     CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));
        
    //     CHECK_CUDA_ERROR(cudaMemcpy(slice.data(), d_slice, slice_size * sizeof(float), cudaMemcpyDeviceToHost));
        
    //     cudaFree(d_slice);
    // }
    
    // void extract_dilated_slice(const Eigen::VectorXi& indices, std::vector<float>& slice, int radius) {
    //     int min_x = indices[0];
    //     int max_x = indices[1];
    //     int min_y = indices[2];
    //     int max_y = indices[3];
    //     int min_z = indices[4];
    //     int max_z = indices[5];
        
    //     size_t slice_size_x = max_x - min_x + 1;
    //     size_t slice_size_y = max_y - min_y + 1;
        
    //     size_t slice_size = slice_size_x * slice_size_y;
        
    //     slice.resize(slice_size);
        
    //     float* d_slice;
        
    //     CHECK_CUDA_ERROR(cudaMalloc(&d_slice, slice_size * sizeof(float)));
    //     CHECK_CUDA_ERROR(cudaMemset(d_slice, -1.0, slice_size * sizeof(float)));
        
    //     launch_extract_dilated_2d_slice_kernel(d_voxel_grid_, d_slice, min_x, max_x, min_y, max_y, min_z, max_z, radius, stream_);
        
    //     CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));
        
    //     CHECK_CUDA_ERROR(cudaMemcpy(slice.data(), d_slice, slice_size * sizeof(float), cudaMemcpyDeviceToHost));
        
    //     cudaFree(d_slice);
    // }
    
    // void extract_esdf(const Eigen::VectorXi& indices, std::vector<float>& esdf) {
    //     int min_x = indices[0];
    //     int max_x = indices[1];
    //     int min_y = indices[2];
    //     int max_y = indices[3];
    //     int min_z = indices[4];
    //     int max_z = indices[5];
        
    //     size_t esdf_size_x = max_x - min_x + 1;
    //     size_t esdf_size_y = max_y - min_y + 1;
        
    //     size_t esdf_size = esdf_size_x * esdf_size_y;
        
    //     esdf.resize(esdf_size);
        
    //     float* d_binary_slice;
        
    //     CHECK_CUDA_ERROR(cudaMalloc(&d_binary_slice, esdf_size * sizeof(float)));
    //     launch_initialize_float_kernel(d_binary_slice, FLT_MAX, esdf_size);
        
    //     launch_extract_binary_slice_kernel(d_voxel_grid_, d_binary_slice, min_x, max_x, min_y, max_y, min_z, max_z, stream_);
        
    //     float* d_esdf;
        
    //     CHECK_CUDA_ERROR(cudaMalloc(&d_esdf, esdf_size * sizeof(float)));
        
    //     CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));
        
    //     launch_edt_kernels(d_binary_slice, d_esdf, esdf_size_x, esdf_size_y, stream_);
        
    //     CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));
        
    //     CHECK_CUDA_ERROR(cudaMemcpy(esdf.data(), d_esdf, esdf_size * sizeof(float), cudaMemcpyDeviceToHost));
        
    //     cudaFree(d_binary_slice);
    //     cudaFree(d_esdf);
    // }



VoxelMapping::VoxelMapping(size_t map_update_capacity, size_t voxel_update_capacity, float resolution, float min_depth, float max_depth, VoxelType log_odds_occupied, VoxelType log_odds_free, VoxelType log_odds_min, VoxelType log_odds_max, VoxelType occupancy_threshold, VoxelType free_threshold)
: pimpl_(std::make_unique<VoxelMappingImpl>(map_update_capacity, voxel_update_capacity, resolution, min_depth, max_depth, log_odds_occupied, log_odds_free, log_odds_min, log_odds_max, occupancy_threshold, free_threshold))
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

std::vector<VoxelType> VoxelMapping::get_3d_block(const std::vector<VoxelType>& aabb) {
    return pimpl_->voxel_map_->extract_grid_block(aabb.data());
}

// std::vector<float> VoxelMapping::get_grid_block(const Eigen::VectorXi& aabb_indices) {
//     // return pimpl_->get_grid_block(aabb_indices);
//     return {}; // Placeholder for actual implementation
// }

// void VoxelMapping::extract_slice(const Eigen::VectorXi& indices, std::vector<float>& slice) {
//     pimpl_->extract_slice(indices, slice);
// }
// void VoxelMapping::extract_dilated_slice(const Eigen::VectorXi& indices, std::vector<float>& slice, int radius) {
//     pimpl_->extract_dilated_slice(indices, slice, radius);
// }
// void VoxelMapping::extract_esdf(const Eigen::VectorXi& indices, std::vector<float>& esdf) {
//     pimpl_->extract_esdf(indices, esdf);
// }