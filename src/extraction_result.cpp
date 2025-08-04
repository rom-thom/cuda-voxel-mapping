#include "voxel-mapping/extraction_result.hpp"
#include "voxel-mapping/extraction_result_impl.hpp"
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <string>

namespace voxel_mapping {

ExtractionResult::~ExtractionResult() = default;
ExtractionResult::ExtractionResult(ExtractionResult&&) = default;
ExtractionResult& ExtractionResult::operator=(ExtractionResult&&) = default;

ExtractionResult::ExtractionResult() : pimpl_(nullptr) {}

void ExtractionResult::wait() {
    if (pimpl_) {
        pimpl_->wait();
    }
}

size_t ExtractionResult::size_bytes() const {
    return pimpl_ ? pimpl_->size_bytes() : 0;
}

template<typename T>
const T* ExtractionResult::data() const {
    if (!pimpl_) {
        return nullptr;
    }
    if (pimpl_->type() != typeid(T)) {
        spdlog::error("Type mismatch: expected {}, got {}", typeid(T).name(), pimpl_->type().name());
        throw std::runtime_error(std::string("Type mismatch: ExtractionResult contains ") +
                                 std::string(pimpl_->type().name()) +
                                 " but requested type " +
                                 std::string(typeid(T).name()));
    }
    return static_cast<const T*>(pimpl_->data());
}

template const int* ExtractionResult::data<int>() const;

} // namespace voxel_mapping