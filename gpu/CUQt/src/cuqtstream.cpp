#include "cuqtstream.h"

#include <type_traits>
#include <cuda_runtime.h>
#include <mutex>
#include <QTimer>

#include <QDebug>

namespace
{
struct HostCallbackData
{
    std::function<void()> cb;
};
} // namespace

class CUQtStream::Priv final
{
public:
    cudaStream_t handle{nullptr};
    std::mutex mutex;
    std::vector<std::unique_ptr<HostCallbackData>> hostCallbacks;
};

static void CUDART_CB hostCallback(void *userData)
{
    auto cbData{static_cast<HostCallbackData*>(userData)};
    if (cbData && cbData->cb)
    {
        cbData->cb();
    }
}

CUQtStream::CUQtStream(CreationFlags flags, QObject *parent)
    : QObject{parent}
    , priv_{new Priv()}
{
    CUQt::discardLastError();
    cudaStreamCreateWithFlags(&priv_->handle,
                              static_cast<std::underlying_type<CreationFlags>::type>(flags));
}

CUQtStream::~CUQtStream()
{
    if (priv_->handle)
    {
        cudaStreamDestroy(priv_->handle);
    }
}

CUQtStream::CompletionStatus CUQtStream::status() const noexcept
{
    const auto code{cudaStreamQuery(priv_->handle)};
    if (code == cudaSuccess)
    {
        return CompletionStatus::Done;
    }
    else if (code == cudaErrorNotReady)
    {
        return CompletionStatus::NotReady;
    }
    else
    {
        return CompletionStatus::Unknown;
    }
}

void CUQtStream::synchronize()
{
    cudaStreamSynchronize(priv_->handle);
}

CUQtStream::operator cudaStream_t() const noexcept
{
    return priv_->handle;
}

cudaError CUQtStream::enqueueFunction(const std::function<void()> &fn)
{
    if (!fn)
    {
        return cudaErrorInvalidValue;
    }

    auto cbData{std::make_unique<HostCallbackData>()};
    cbData->cb = [this, fn, ptr = cbData.get()]()
    {
        fn();
        std::lock_guard<std::mutex> lock{priv_->mutex};
        for (auto it = priv_->hostCallbacks.begin(); it != priv_->hostCallbacks.end(); ++it)
        {
            if (it->get() == ptr)
            {
                priv_->hostCallbacks.erase(it);
                break;
            }
        }
    };

    auto returnCode{cudaLaunchHostFunc(priv_->handle, hostCallback, cbData.get())};
    if (returnCode == cudaSuccess)
    {
        std::lock_guard<std::mutex> lock{priv_->mutex};
        priv_->hostCallbacks.push_back(std::move(cbData));
    }
    return returnCode;
}
