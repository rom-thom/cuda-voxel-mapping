#ifndef EXTRACTOR_CUH
#define EXTRACTOR_CUH

#include <cuda_runtime.h>
#include "voxel-mapping/types.hpp"

namespace voxel_mapping {

class GridProcessor {
public:
    /**
     * @brief Constructor for the GridProcessor responsible for processing extracted voxel blocks before the results are copied to the host.
     * @param occupancy_threshold Threshold for occupancy.
     * @param free_threshold Threshold for free space.
     */
    GridProcessor(int occupancy_threshold, int free_threshold);

    /**
     * @brief Launches the binary slice extraction kernel that converts an AABB into a binary slice.
     * Sets occupied voxels to 0 and the rest to the greater of size_x or size_y.
     * @param d_aabb Pointer to the device memory containing the AABB data.
     * @param d_slice Pointer to the device memory where the extracted slice will be stored.
     * @param min_x Minimum x-coordinate of the AABB.
     * @param min_y Minimum y-coordinate of the AABB.
     * @param min_z Minimum z-coordinate of the AABB.
     * @param size_x size of the slice.
     * @param size_y size of the slice.
     * @param size_z size of the slice to use for projection to the 2D slice.
     * @param stream CUDA stream for asynchronous execution.
     */
    void launch_extract_binary_slice_kernel(
    const VoxelType* d_aabb, int* d_slice,
    int min_x, int min_y,int min_z, 
    int size_x, int size_y, int size_z,
    cudaStream_t stream);

    /**
     * @brief Launches the EDT (Euclidean Distance Transform) kernels.
     * @param d_binary_slice Pointer to the device memory containing the binary slice.
     * @param d_edt Pointer to the device memory where the EDT slice will be stored.
     * @param size_x Width of the binary slice.
     * @param size_y Height of the binary slice.
     * @param stream CUDA stream for asynchronous execution.
     */
    void launch_edt_kernels(int* d_binary_slice, int* d_edt, int size_x, int size_y, cudaStream_t stream);

private:
    int occupancy_threshold_;
    int free_threshold_;
};

} // namespace voxel_mapping

#endif // EXTRACTOR_CUH