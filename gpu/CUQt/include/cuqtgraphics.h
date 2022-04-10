#ifndef CUQTGRAPHICS_H
#define CUQTGRAPHICS_H

#include "cuqt.h"

#include <QImage>
#include <QScopedPointer>

namespace CUQt
{
CUQT_DLL_SPECS cudaChannelFormatDesc toCudaChannelFormat(QImage::Format image_format) noexcept;
} // namespace CUQt

class CUQT_DLL_SPECS CUQtTexture final
{
public:
    static bool isFormatSupported(const QImage::Format format) noexcept;

    CUQtTexture();
    CUQtTexture(const QImage &image, cudaStream_t stream = 0);
    ~CUQtTexture();

    cudaTextureDesc textureDescriptor() const noexcept;
    void setTextureDescriptor(const cudaTextureDesc &tex_desc) noexcept;

    bool isValid() const noexcept;

    cudaTextureObject_t handle() const noexcept;
    void *devicePointer() const noexcept;
    size_t pitch() const noexcept;
    cudaChannelFormatDesc formatDescriptor() const noexcept;

    cudaError preallocate(size_t width, size_t height, QImage::Format format, cudaStream_t stream = 0);

    cudaError upload(const QImage &image, cudaStream_t stream = 0);

    CUQtResult<QImage> download(cudaStream_t stream = 0) const;
    cudaError download(QImage &target, cudaStream_t stream = 0) const;

private:
    class Priv;
    QScopedPointer<Priv> priv_;
};

#endif // CUQTGRAPHICS_H
