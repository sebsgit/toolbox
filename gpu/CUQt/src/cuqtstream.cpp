#include "cuqtstream.h"

#include <type_traits>
#include <cuda_runtime.h>
#include <QTimer>

#include <QDebug>

class CUQtStream::Priv final
{
public:
    cudaStream_t handle{nullptr};
};

static void streamCompletedCallback(cudaStream_t stream,  cudaError_t status, void* userData)
{
    CUQtStream *unwrapped{reinterpret_cast<CUQtStream*>(userData)};
    QTimer::singleShot(0, unwrapped, &CUQtStream::completed);
}

CUQtStream::CUQtStream(CreationFlags flags, QObject *parent)
    : QObject{parent}
    , priv_{new Priv()}
{
    CUQt::discardLastError();
    cudaStreamCreateWithFlags(&priv_->handle,
                              static_cast<std::underlying_type<CreationFlags>::type>(flags));
    if (priv_->handle)
    {
        cudaStreamAddCallback(priv_->handle, streamCompletedCallback, this, 0);
    }
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
