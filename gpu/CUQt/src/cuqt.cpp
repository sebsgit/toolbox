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
