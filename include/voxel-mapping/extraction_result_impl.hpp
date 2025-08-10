#ifndef EXTRACTION_RESULT_IMPL_HPP
#define EXTRACTION_RESULT_IMPL_HPP

#include "voxel-mapping/extraction_result.hpp"
#include <cuda_runtime.h>
#include <typeinfo>

namespace voxel_mapping {

class ExtractionResultBase {
public:
    virtual ~ExtractionResultBase() = default;
    virtual void wait() = 0;
    virtual const void* data() const = 0;
    virtual size_t size_bytes() const = 0;
    virtual const std::type_info& type() const = 0;
};

template<typename T>
class ExtractionResultTyped : public ExtractionResultBase {
public:
    ~ExtractionResultTyped() {
        if (event_) cudaEventDestroy(event_);
        if (h_pinned_data_) cudaFreeHost(h_pinned_data_);
        if (d_data_) cudaFree(d_data_);
    }

    void wait() override {
        if (event_) cudaEventSynchronize(event_);
    }
    const void* data() const override { return h_pinned_data_; }
    size_t size_bytes() const override { return size_bytes_; }
    const std::type_info& type() const override { return typeid(T); }

    void* h_pinned_data_ = nullptr;
    void* d_data_ = nullptr;
    cudaEvent_t event_ = nullptr;
    size_t size_bytes_ = 0;
};

} // namespace voxel_mapping

#endif // EXTRACTION_RESULT_IMPL_HPP
