#include "cuqtdevice.h"

class CUQtDevice::Priv final
{
public:
    int32_t dev_id{-1};
};

size_t CUQtDevice::numberOfDevices()
{
    int32_t dev_count{};
    cudaGetDeviceCount(&dev_count);
    return static_cast<size_t>(dev_count);
}

CUQtDevice::CUQtDevice(QObject *parent)
    : QObject{parent}
    , priv_{new Priv}
{
    CUQt::discardLastError();
    cudaGetDevice(&priv_->dev_id);
}

CUQtDevice::~CUQtDevice() = default;

void CUQtDevice::setId(const int32_t id) noexcept
{
    if (priv_->dev_id != id)
    {
        const auto err_code{cudaSetDevice(id)};
        if (err_code == cudaSuccess)
        {
            priv_->dev_id = id;
        }
    }
}

void CUQtDevice::synchronize()
{
    cudaDeviceSynchronize();
}

int32_t CUQtDevice::id() const noexcept
{
    return priv_->dev_id;
}

cudaDeviceProp CUQtDevice::properties() const noexcept
{
    cudaDeviceProp result{};
    cudaGetDeviceProperties(&result, priv_->dev_id);
    return result;
}

QByteArray CUQtDevice::pciBusId() const noexcept
{
    QByteArray result{16, ' '};
    const auto err_code{cudaDeviceGetPCIBusId(result.data(), result.length(), priv_->dev_id)};
    if (CUQt::isError(err_code))
    {
        return {};
    }
    return result.trimmed();
}

