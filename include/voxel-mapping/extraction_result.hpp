#ifndef EXTRACTION_RESULT_HPP
#define EXTRACTION_RESULT_HPP

#include <memory>
#include <cstddef>
#include <typeinfo>

namespace voxel_mapping {

class ExtractionResultBase;
class VoxelMappingImpl;

/**
 * @brief A non-template handle to an asynchronous data extraction result.
 */
class ExtractionResult {
public:
    ~ExtractionResult();
    ExtractionResult(ExtractionResult&&);
    ExtractionResult& operator=(ExtractionResult&&);

    /**
     * @brief Blocks the calling thread until the GPU has finished this extraction.
     */
    void wait();

    /**
     * @brief Gets the size of the result data in bytes.
     * @return The size of the data buffer in bytes.
     */
    size_t size_bytes() const;

    /**
     * @brief Gets a typed pointer to the result data on the host.
     * @warning You must call wait() before calling this function.
     * @return A const pointer to the data, throwing an exception if the type does not match.
     */
    template<typename T>
    const T* data() const;

private:
    friend class VoxelMappingImpl;
    ExtractionResult();

    std::unique_ptr<ExtractionResultBase> pimpl_;
};

} // namespace voxel_mapping
#endif // EXTRACTION_RESULT_HPP
