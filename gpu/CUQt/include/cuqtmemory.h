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
    MemcpyInvalidDirection = cudaErrorInvalidMemcpyDirection
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
        return CUQt::MemcpyResult::MemcpyInvalidValue;
    }

    CUQt::MemcpyResult download(T* destination, size_t n_elem, size_t offset_in_source = 0) noexcept
    {
        const auto err{cudaMemcpy(destination, ptr_ + offset_in_source, n_elem * sizeof(T), cudaMemcpyDeviceToHost)};
        if (QMetaEnum::fromType<CUQt::MemcpyResult>().valueToKey(err))
        {
            return static_cast<CUQt::MemcpyResult>(err);
        }
        return CUQt::MemcpyResult::MemcpyInvalidValue;
    }

    const T* devicePointer() const noexcept { return ptr_; }

    T* devicePointer() noexcept { return ptr_; }

private:
    T *ptr_{nullptr};
};

#endif // CUQTMEMORY_H
