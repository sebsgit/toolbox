#ifndef CUQTMEMORY_H
#define CUQTMEMORY_H

#include <QObject>
#include <QMetaEnum>
#include "cuqt.h"

namespace CUQt
{
Q_NAMESPACE_EXPORT(CUQT_DLL_SPECS);

enum MemcpyResult : uint8_t
{
    MemcpySuccess = cudaSuccess,
    MemcpyInvalidValue = cudaErrorInvalidValue,
    MemcpyInvalidPitchValue = cudaErrorInvalidPitchValue,
    MemcpyInvalidDirection = cudaErrorInvalidMemcpyDirection,
    MemcpyUnknownError = 0xff
};

Q_ENUM_NS(MemcpyResult);
} // namespace CUQt

template <typename T>
class CUQtDeviceMemoryBlock final
{
public:
    explicit CUQtDeviceMemoryBlock(size_t num_elements) noexcept
    {
        const auto err{cudaMalloc(&ptr_, num_elements * sizeof(T))};
        if (err != cudaSuccess)
        {
            ptr_ = nullptr;
        }
    }

    ~CUQtDeviceMemoryBlock() noexcept
    {
        if (ptr_)
        {
            cudaFree(ptr_);
        }
    }

    CUQtDeviceMemoryBlock(const CUQtDeviceMemoryBlock &) = delete;
    CUQtDeviceMemoryBlock& operator=(const CUQtDeviceMemoryBlock &) = delete;

    CUQtDeviceMemoryBlock(CUQtDeviceMemoryBlock &&other) noexcept:
          ptr_{other.ptr_}
    {
        other.ptr_ = nullptr;
    }

    CUQtDeviceMemoryBlock& operator=(CUQtDeviceMemoryBlock && other) noexcept
    {
        if (this != &other)
        {
            if (ptr_)
            {
                cudaFree(ptr_);
            }
            ptr_ = other.ptr_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    bool isValid() const noexcept
    {
        return ptr_ != nullptr;
    }

    CUQt::MemcpyResult upload(const T* source, size_t n_elem, size_t offset_in_target = 0) noexcept
    {
        const auto err{cudaMemcpy(ptr_ + offset_in_target, source, n_elem * sizeof(T), cudaMemcpyHostToDevice)};
        if (QMetaEnum::fromType<CUQt::MemcpyResult>().valueToKey(err))
        {
            return static_cast<CUQt::MemcpyResult>(err);
        }
        return CUQt::MemcpyResult::MemcpyUnknownError;
    }

    CUQt::MemcpyResult download(T* destination, size_t n_elem, size_t offset_in_source = 0) noexcept
    {
        const auto err{cudaMemcpy(destination, ptr_ + offset_in_source, n_elem * sizeof(T), cudaMemcpyDeviceToHost)};
        if (QMetaEnum::fromType<CUQt::MemcpyResult>().valueToKey(err))
        {
            return static_cast<CUQt::MemcpyResult>(err);
        }
        return CUQt::MemcpyResult::MemcpyUnknownError;
    }

    const T* devicePointer() const noexcept { return ptr_; }

    T* devicePointer() noexcept { return ptr_; }

private:
    T *ptr_{nullptr};
};

template <typename T>
class CUQtDeviceMemoryBlock2D final
{
public:
    explicit CUQtDeviceMemoryBlock2D(size_t width, size_t height) noexcept
    {
        const auto err_code{cudaMallocPitch(&ptr_, &pitch_, width * sizeof(T), height)};
        if (err_code != cudaSuccess)
        {
            ptr_ = nullptr;
            pitch_ = 0;
        }
    }

    ~CUQtDeviceMemoryBlock2D() noexcept
    {
        if (ptr_)
        {
            cudaFree(ptr_);
        }
    }

    CUQtDeviceMemoryBlock2D(const CUQtDeviceMemoryBlock2D &) = delete;
    CUQtDeviceMemoryBlock2D& operator=(const CUQtDeviceMemoryBlock2D &) = delete;

    CUQtDeviceMemoryBlock2D(CUQtDeviceMemoryBlock2D &&other) noexcept:
          ptr_{other.ptr_},
          pitch_{other.pitch_}
    {
        other.ptr_ = nullptr;
        other.pitch_ = 0;
    }

    CUQtDeviceMemoryBlock2D& operator=(CUQtDeviceMemoryBlock2D && other) noexcept
    {
        if (this != &other)
        {
            if (ptr_)
            {
                cudaFree(ptr_);
            }
            ptr_ = other.ptr_;
            pitch_ = other.pitch_;
            other.ptr_ = nullptr;
            other.pitch_ = 0;
        }
        return *this;
    }

    bool isValid() const noexcept
    {
        return ptr_ != nullptr;
    }

    size_t pitch() const noexcept
    {
        return pitch_;
    }

    const T* devicePointer() const noexcept { return ptr_; }

    T* devicePointer() noexcept { return ptr_; }

    CUQt::MemcpyResult upload(const T* source, size_t source_pitch_in_bytes, size_t width, size_t height) noexcept
    {
        const auto err{cudaMemcpy2D(ptr_, pitch_, source, source_pitch_in_bytes, width * sizeof(T), height, cudaMemcpyHostToDevice)};
        if (QMetaEnum::fromType<CUQt::MemcpyResult>().valueToKey(err))
        {
            return static_cast<CUQt::MemcpyResult>(err);
        }
        return CUQt::MemcpyResult::MemcpyUnknownError;
    }

    CUQt::MemcpyResult download(T* target, size_t target_pitch_in_bytes, size_t width, size_t height) noexcept
    {
        const auto err{cudaMemcpy2D(target, target_pitch_in_bytes, ptr_, pitch_, width * sizeof(T), height, cudaMemcpyDeviceToHost)};
        if (QMetaEnum::fromType<CUQt::MemcpyResult>().valueToKey(err))
        {
            return static_cast<CUQt::MemcpyResult>(err);
        }
        return CUQt::MemcpyResult::MemcpyUnknownError;
    }

private:
    T *ptr_{nullptr};
    size_t pitch_{0};
};

#endif // CUQTMEMORY_H
