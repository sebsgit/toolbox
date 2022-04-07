#ifndef CUQT_H
#define CUQT_H

#include <QtGlobal>

#include <cuda_runtime.h>

#ifdef CUQT_LIBRARY
#define CUQT_DLL_SPECS Q_DECL_EXPORT
#else
#define CUQT_DLL_SPECS Q_DECL_IMPORT
#endif

class QDebug;

extern CUQT_DLL_SPECS QDebug operator<<(QDebug, cudaDeviceAttr);
extern CUQT_DLL_SPECS QDebug operator<<(QDebug, cudaError);
extern CUQT_DLL_SPECS QDebug operator<<(QDebug, cudaResourceType);
extern CUQT_DLL_SPECS QDebug operator<<(QDebug, cudaChannelFormatKind);
extern CUQT_DLL_SPECS QDebug operator<<(QDebug, const dim3 &);

template <typename T>
struct CUQT_DLL_SPECS CUQtResult final
{
    T value{};
    cudaError status{cudaErrorInvalidValue};
};

namespace CUQt {

struct Version final
{
    int32_t major{};
    int32_t minor{};
};

CUQT_DLL_SPECS Version CUDAVersion() noexcept;
CUQT_DLL_SPECS Version CUDADriverVersion() noexcept;
CUQT_DLL_SPECS bool hasError() noexcept;
CUQT_DLL_SPECS bool isError(cudaError err_code) noexcept;
CUQT_DLL_SPECS QString errorDescription(cudaError err_code) noexcept;
CUQT_DLL_SPECS cudaError lastError() noexcept;
CUQT_DLL_SPECS QString lastErrorDescription() noexcept;
CUQT_DLL_SPECS void discardLastError() noexcept;
} // namespace CUQt

CUQT_DLL_SPECS QDebug operator<<(QDebug, const CUQt::Version &);

#endif // CUQT_H
