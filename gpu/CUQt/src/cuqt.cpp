#include "cuqt.h"

#include <QString>

namespace CUQt
{
Version CUDAVersion() noexcept
{
    int32_t result{};
    cudaRuntimeGetVersion(&result);
    return Version{result / 1000, (result % 100) / 10};
}

Version CUDADriverVersion() noexcept
{
    int32_t result{};
    cudaDriverGetVersion(&result);
    return Version{result / 1000, (result % 100) / 10};
}

bool hasError() noexcept
{
    return isError(lastError());
}

bool isError(cudaError err_code) noexcept
{
    return err_code != cudaSuccess;
}

QString errorDescription(cudaError err_code) noexcept
{
    return QString::fromLocal8Bit(cudaGetErrorString(err_code));
}

cudaError lastError() noexcept
{
    return cudaPeekAtLastError();
}

QString lastErrorDescription() noexcept
{
    return errorDescription(lastError());
}

void discardLastError() noexcept
{
    Q_UNUSED(cudaGetLastError());
}

} // namespace CUQt

bool operator==(const cudaChannelFormatDesc &f1, const cudaChannelFormatDesc &f2) noexcept
{
    return f1.f == f2.f && f1.w == f2.w && f1.x == f2.x && f1.y == f2.y && f1.z == f2.z;
}

bool operator!=(const cudaChannelFormatDesc &f1, const cudaChannelFormatDesc &f2) noexcept
{
    return !(f1 == f2);
}
