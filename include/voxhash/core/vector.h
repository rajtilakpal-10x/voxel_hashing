#pragma once

#include "voxhash/core/cuda_utils.h"
#include "voxhash/core/types.h"
#include <memory>

namespace voxhash
{

    template <typename DataType>
    class Vector
    {
    public:
        using Ptr = std::unique_ptr<Vector<DataType>>;

        Vector(size_t size, MemoryType type);
        virtual ~Vector();

        // Disable copy
        Vector(const Vector &) = delete;
        Vector &operator=(const Vector &) = delete;

        Vector(Vector &&) noexcept;
        Vector &operator=(Vector &&) noexcept;

        DataType *data() const;

        size_t size() const;

        MemoryType location() const;

        // Vector<DataType> clone(const CudaStream &cuda_stream = CudaStreamOwning());

        DataType *release();

        bool valid() const;

        static std::unique_ptr<Vector<DataType>> copyFrom(const Vector<DataType> &src, const MemoryType target_type,
                                                          const CudaStream &stream = CudaStreamOwning());

    protected:
        DataType *ptr_{nullptr};
        MemoryType type_{MemoryType::kHost};
        size_t size_{0};
        bool is_valid_{false};
        void copyFromImpl(const Vector<DataType> &src,
                          const CudaStream &stream);
        virtual void allocateImpl(const CudaStream &cuda_stream = CudaStreamOwning());
        std::unique_ptr<Vector<DataType>> createEmpty(MemoryType target_type) const;
        void allocate(const CudaStream &cuda_stream = CudaStreamOwning());
        void free(const CudaStream &cuda_stream = CudaStreamOwning());
    };

} // namespace voxhash