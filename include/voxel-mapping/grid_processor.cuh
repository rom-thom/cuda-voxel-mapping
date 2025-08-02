#ifndef EXTRACTOR_CUH
#define EXTRACTOR_CUH

#include <cuda_runtime.h>
#include "voxel-mapping/internal_types.cuh"

namespace voxel_mapping {

enum class Dimension { X, Y, Z };

class GridProcessor {
public:
    /**
     * @brief Constructor for the GridProcessor responsible for processing extracted voxel blocks before the results are copied to the host.
     * @param occupancy_threshold Threshold for occupancy.
     * @param free_threshold Threshold for free space.
     */
    GridProcessor(int occupancy_threshold, int free_threshold, int edt_max_distance);
    
    /**
     * @brief Launches the kernel for performing 2d Euclidean Distance Transform (EDT) on a set of slices.
     * @param d_edt_slices Pointer to the device memory where the EDT slices will be stored.
     * @param size_x Size of the grid in the X dimension.
     * @param size_y Size of the grid in the Y dimension.
     * @param num_slices Number of slices to extract.
     * @param stream CUDA stream for asynchronous execution.
     */
    void launch_edt_slice_kernels(int* d_edt_slices, int size_x, int size_y, int num_slices, cudaStream_t stream);

    /**
     * @brief Launches the kernel for performing 3d Euclidean Distance Transform (EDT) on a block of voxels.
     * @param d_edt_block Pointer to the device memory where the EDT block will be stored.
     * @param size_x Size of the grid in the X dimension.
     * @param size_y Size of the grid in the Y dimension.
     * @param size_z Size of the grid in the Z dimension.
     * @param stream CUDA stream for asynchronous execution.
     */
    void launch_3d_edt_kernels(int* d_edt_block, int size_x, int size_y, int size_z, cudaStream_t stream);
    
private:
    /**
     * @brief Launches the appropriate kernel for extracting a block of voxels based on the specified extraction type.
     * @param d_grid Pointer to the device memory where the voxel grid is stored.
     * @param size_x Size of the grid in the X dimension.
     * @param size_y Size of the grid in the Y dimension.
     * @param size_z Size of the grid in the Z dimension.
     * @param stream CUDA stream for asynchronous execution.
     */
    template <ExtractionType Type>
    void launch_edt_kernels_internal(int* d_grid, int size_x, int size_y, int size_z, cudaStream_t stream);

    int occupancy_threshold_;
    int free_threshold_;
};

} // namespace voxel_mapping

#endif // EXTRACTOR_CUH