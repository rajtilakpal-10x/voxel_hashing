#include <iostream>

#include "voxhash/core/vector.h"
#include "voxhash/core/voxels.h"

namespace voxhash {

template <typename DataType>
Vector<DataType>::Vector(size_t size, MemoryType type) : size_(size), type_(type) {
    allocate();
}

template <typename DataType>
void Vector<DataType>::allocateImpl(const CudaStream& cuda_stream) {
    if (type_ == MemoryType::kUnified) {
        CUDA_CHECK(cudaMallocManaged(&ptr_, sizeof(DataType) * size_, cudaMemAttachGlobal));
    } else if (type_ == MemoryType::kDevice) {
        CUDA_CHECK(cudaMallocAsync(&ptr_, sizeof(DataType) * size_, cuda_stream));
    } else {
        CUDA_CHECK(cudaMallocHost(&ptr_, sizeof(DataType) * size_));
    }
}

template <typename DataType>
void Vector<DataType>::allocate(const CudaStream& cuda_stream) {
    if (is_valid_) free();
    allocateImpl(cuda_stream);
    is_valid_ = true;
}

template <typename DataType>
void Vector<DataType>::free(const CudaStream& cuda_stream) {
    if (ptr_ != nullptr) {
        // std::cout << "Freeing\n";
        if (type_ == MemoryType::kHost) {
            CUDA_CHECK(cudaFreeHost(ptr_));
        } else if (type_ == MemoryType::kDevice) {
            CUDA_CHECK(cudaFreeAsync(ptr_, cuda_stream));
        } else {
            CUDA_CHECK(cudaFree(ptr_));
        }
        ptr_ = nullptr;
        is_valid_ = false;
    }
}

template <typename DataType>
DataType* Vector<DataType>::data() const {
    return ptr_;
}

template <typename DataType>
size_t Vector<DataType>::size() const {
    return size_;
}

template <typename DataType>
MemoryType Vector<DataType>::location() const {
    return type_;
}

template <typename DataType>
bool Vector<DataType>::valid() const {
    return is_valid_;
}

template <typename DataType>
DataType* Vector<DataType>::release() {
    DataType* ptr = ptr_;
    ptr_ = nullptr;
    is_valid_ = false;
    return ptr;
}

template <typename DataType>
Vector<DataType>::~Vector() {
    try {
        if (is_valid_) free();
    } catch (const CudaError& e) {
    }
}

template <typename DataType>
std::shared_ptr<Vector<DataType>> Vector<DataType>::copyFrom(
        const Vector<DataType>& src, const MemoryType target_type, const CudaStream& stream) {
    Ptr dst = src.createEmpty(target_type);

    dst->copyFromImpl(src, stream);
    return dst;
}

template <typename DataType>
std::shared_ptr<Vector<DataType>> Vector<DataType>::copyFrom(
        const std::vector<DataType>& src, const MemoryType target_type, const CudaStream& stream) {
    Ptr dst = std::make_shared<Vector<DataType>>(src.size(), target_type);

    CUDA_CHECK(cudaMemcpyAsync(
            dst->data(), src.data(), sizeof(DataType) * src.size(), cudaMemcpyDefault, stream));
    return dst;
}

template <typename DataType>
bool Vector<DataType>::setFrom(const Vector<DataType>& src, const CudaStream& cuda_stream) {
    if (src.size() != size_) return false;

    CUDA_CHECK(cudaMemcpyAsync(
            this->ptr_, src.ptr_, sizeof(DataType) * size_, cudaMemcpyDefault, cuda_stream));
    return true;
}

template <typename DataType>
std::shared_ptr<Vector<DataType>> Vector<DataType>::createEmpty(MemoryType target_type) const {
    return std::make_shared<Vector<DataType>>(this->size_, target_type);
}

template <typename DataType>
void Vector<DataType>::copyFromImpl(const Vector<DataType>& src, const CudaStream& stream) {
    CUDA_CHECK(
            cudaMemcpyAsync(ptr_, src.ptr_, sizeof(DataType) * size_, cudaMemcpyDefault, stream));
}

template <typename DataType>
Vector<DataType>::Vector(Vector&& other) noexcept {
    ptr_ = other.ptr_;
    size_ = other.size_;
    type_ = other.type_;
    is_valid_ = other.is_valid_;

    other.ptr_ = nullptr;
    other.is_valid_ = false;
    other.size_ = 0;
}

template <typename DataType>
Vector<DataType>& Vector<DataType>::operator=(Vector&& other) noexcept {
    if (this != &other) {
        if (is_valid_) free();

        ptr_ = other.ptr_;
        size_ = other.size_;
        type_ = other.type_;
        is_valid_ = other.is_valid_;

        other.ptr_ = nullptr;
        other.is_valid_ = false;
        other.size_ = 0;
    }
    return *this;
}

template <typename DataType>
DataType& Vector<DataType>::operator[](size_t idx) const {
    return this->ptr_[idx];
}

template <typename DataType>
void Vector<DataType>::clear(const CudaStream& cuda_stream) {
    CUDA_CHECK(cudaMemsetAsync(this->ptr_, 0, sizeof(DataType) * this->size_, cuda_stream));
}

template class Vector<int>;
template class Vector<float>;
template class Vector<TsdfVoxel>;
template class Vector<SemanticVoxel>;

template class Vector<Index3D>;
template class Vector<Vector3f>;
template class Vector<bool>;
template class Vector<Bool>;

template class Vector<int*>;
template class Vector<float*>;
template class Vector<TsdfVoxel*>;
template class Vector<SemanticVoxel*>;
}  // namespace voxhash