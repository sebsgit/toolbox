#include "cuqtevent.h"
#include <type_traits>
#include <QMetaEnum>

class CUQtEvent::Priv final
{
public:
    cudaEvent_t handle{nullptr};
};

CUQtEvent::CUQtEvent(CUQtEvent::CreationFlags flags, QObject *parent)
    : QObject{parent}
    , priv_{new Priv()}
{
    cudaEventCreateWithFlags(&priv_->handle, static_cast<uint32_t>(flags));
}

CUQtEvent::~CUQtEvent()
{
    if (priv_->handle)
    {
        cudaEventDestroy(priv_->handle);
    }
}

CUQtEvent::operator cudaEvent_t() const noexcept
{
    return priv_->handle;
}

std::chrono::milliseconds CUQtEvent::elapsedTime(const cudaEvent_t to) const noexcept
{
    float ms{0.0f};
    const auto err{cudaEventElapsedTime(&ms, priv_->handle, to)};
    if (err == cudaSuccess)
    {
        return std::chrono::milliseconds(static_cast<int64_t>(ms));
    }
    return {};
}

CUQtEvent::CompletionStatus CUQtEvent::status() const noexcept
{
    const auto raw_err_code{cudaEventQuery(priv_->handle)};
    if (QMetaEnum::fromType<CUQtEvent::CompletionStatus>().valueToKey(raw_err_code))
    {
        return static_cast<CompletionStatus>(raw_err_code);
    }
    return CompletionStatusInvalidValue;
}

void CUQtEvent::synchronize()
{
    cudaEventSynchronize(priv_->handle);
}

CUQtEvent::RecordStatus CUQtEvent::record(cudaStream_t stream)
{
    const auto err{cudaEventRecord(priv_->handle, stream)};
    if (QMetaEnum::fromType<CUQtEvent::RecordStatus>().valueToKey(err))
    {
        return static_cast<RecordStatus>(err);
    }
    return RecordStatusInvalidValue;
}
