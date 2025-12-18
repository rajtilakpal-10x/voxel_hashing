#pragma once

// Mostly copied from nvblox

#include <cuda_runtime.h>
#include <memory>

namespace voxhash
{

#define checkCudaErrors(val) \
    check_cuda_error_value((val), #val, __FILE__, __LINE__)

    void check_cuda_error_value(cudaError_t result, char const *const func,
                                const char *const file, int const line);

    void warmupCuda();

    /// Declare enum class for cuda stream type.
    enum class CudaStreamType
    {
        // Default cuda stream that is synchronizing with all the other streams.
        kLegacyDefault,
        // Blocking async stream will be blocked by default stream but async with
        // other async streams.
        kBlocking,
        // Non-blocking stream will be async with default and all the other stream.
        kNonBlocking,
        // Similar to default cuda stream but performs synchronization on per-thread
        // basis. Note that --default-stream per-thread needs to be passed to nvcc.
        kPerThreadDefault,
    };

    /// A thin interface to a CudaStream, it is pure virtual and should not be
    /// instantiated.
    class CudaStream
    {
    public:
        virtual ~CudaStream() = default;

        /// Returns the underlying CUDA stream
        /// @return The raw CUDA stream
        virtual cudaStream_t &get() = 0;
        virtual const cudaStream_t &get() const = 0;

        virtual operator cudaStream_t() = 0;
        virtual operator cudaStream_t() const = 0;

        /// Synchronize the stream
        virtual void synchronize() const = 0;

        // Helper function to create CudaStreamOwning based on various CudaStreamType
        // options.
        static std::shared_ptr<CudaStream> createCudaStream(
            CudaStreamType stream_type);

    protected:
        // Creating a CudaStream instance should never be performed outside of derived
        // class.
        CudaStream() = default;
    };

    /// @brief A default stream implementation to CudaStream, with the option of
    /// passing either the legacy default stream or per-thread default stream.
    class DefaultStream final : public CudaStream
    {
    public:
        DefaultStream(CUstream_st *stream) : default_stream_(stream) {}

        // Note that default stream is owned by system and should not be destroyed.
        virtual ~DefaultStream() = default;

        // Do not allow constructing a DefaultStream object through copying or
        // assignment.
        DefaultStream(const DefaultStream &other) = delete;
        DefaultStream(DefaultStream &&other) = delete;
        DefaultStream &operator=(const DefaultStream &other) = delete;
        DefaultStream &operator=(DefaultStream &&other) = delete;

        /// Returns the underlying CUDA stream
        /// @return The raw CUDA stream
        cudaStream_t &get() { return default_stream_; }
        const cudaStream_t &get() const { return default_stream_; }

        operator cudaStream_t() { return default_stream_; }
        operator cudaStream_t() const { return default_stream_; }

        /// Synchronize the stream. With default stream, synchronization is performed
        /// by device.
        void synchronize() const;

    protected:
        cudaStream_t default_stream_;
    };

    /// @brief A async stream implementation to CudaStream, with the async stream
    /// ownership to be implemented in the derived class.
    class CudaStreamAsync : public CudaStream
    {
    public:
        virtual ~CudaStreamAsync() = default;

        // Copying streams interface
        CudaStreamAsync(const CudaStreamAsync &other) = default;
        CudaStreamAsync(CudaStreamAsync &&other) = default;
        CudaStreamAsync &operator=(const CudaStreamAsync &other) = default;
        CudaStreamAsync &operator=(CudaStreamAsync &&other) = default;

        /// Returns the underlying CUDA stream
        /// @return The raw CUDA stream
        cudaStream_t &get() final { return *stream_ptr_; }
        const cudaStream_t &get() const final { return *stream_ptr_; }

        operator cudaStream_t() final { return *stream_ptr_; }
        operator cudaStream_t() const final { return *stream_ptr_; }

        /// Synchronize the stream
        void synchronize() const final;

    protected:
        CudaStreamAsync(cudaStream_t *stream_ptr) : stream_ptr_(stream_ptr) {}

        cudaStream_t *stream_ptr_;
    };

    /// @brief A simple RAII holder for a cuda stream.
    /// This class setups up a stream on construction and cleans up on destruction.
    /// The destructor synchronizes the stream prior to cleaning up.
    class CudaStreamOwning : public CudaStreamAsync
    {
    public:
        /// Creates the stream on construction and synchronizes+destroys on
        /// destruction.
        ///
        /// @param flags  Stream creation flags from cuda_runtime.h
        CudaStreamOwning(const unsigned int flags = cudaStreamDefault);

        virtual ~CudaStreamOwning();

        // Can't copy owning streams (because both copies would want ownership)
        // NOTE(alexmillane): We *could* implement move operations if that becomes
        // important. For now streams are also un-movable.
        CudaStreamOwning(const CudaStreamOwning &other) = delete;
        CudaStreamOwning(CudaStreamOwning &&other) = delete;

        // Can't assign streams
        // NOTE(alexmillane): We *could* implement move operations if that becomes
        // important. For now streams are also un-movable.
        CudaStreamOwning &operator=(const CudaStreamOwning &other) = delete;
        CudaStreamOwning &operator=(CudaStreamOwning &&other) = delete;

    protected:
        cudaStream_t stream_;
    };
} // namespace voxhash