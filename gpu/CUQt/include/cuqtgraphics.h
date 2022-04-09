#ifndef CUQTGRAPHICS_H
#define CUQTGRAPHICS_H

#include "cuqt.h"

#include <QImage>
#include <QScopedPointer>

namespace CUQt
{
CUQT_DLL_SPECS cudaChannelFormatDesc toCudaChannelFormat(QImage::Format image_format) noexcept;
} // namespace CUQt

class CUQT_DLL_SPECS CUQtTextureSampler final
{
public:
    CUQtTextureSampler();
    ~CUQtTextureSampler();


};

class CUQT_DLL_SPECS CUQtTexture final
{
public:
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

    //TODO add stream parameters to download
    CUQtResult<QImage> download() const;
    cudaError download(QImage &target) const;

private:
    class Priv;
    QScopedPointer<Priv> priv_;
};

#endif // CUQTGRAPHICS_H
