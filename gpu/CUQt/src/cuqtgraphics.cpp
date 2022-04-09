#include "cuqtgraphics.h"
#include "cuqtmemory.h"

namespace CUQt
{
cudaChannelFormatDesc toCudaChannelFormat(QImage::Format image_format) noexcept
{
    cudaChannelFormatDesc result{};
    switch (image_format)
    {
    case QImage::Format_ARGB32:
        result = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
        break;
    case QImage::Format_RGB32:
        result = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKindUnsigned);
        break;
    case QImage::Format_Grayscale8:
        result = cudaCreateChannelDesc(8, 0, 0, 0, cudaChannelFormatKindUnsigned);
        break;
    default:
        result.f = cudaChannelFormatKindNone;
    }
    return result;
}
} // namespace CUQt

class CUQtTexture::Priv final
{
public:
    cudaTextureObject_t handle{0};
    cudaChannelFormatDesc format_desc{};
    cudaTextureDesc texture_desc{};
    int allocated_width{0};
    int allocated_height{0};
    int logical_width{0};
    QImage::Format allocated_format{QImage::Format_Invalid};
    QScopedPointer<CUQtDeviceMemoryBlock2D<uint8_t>> memory;
};

CUQtTexture::CUQtTexture():
      priv_{new Priv()}
{
    priv_->texture_desc.addressMode[0] = cudaAddressModeClamp;
    priv_->texture_desc.addressMode[1] = cudaAddressModeClamp;
    priv_->texture_desc.addressMode[2] = cudaAddressModeClamp;
    priv_->texture_desc.filterMode = cudaFilterModePoint;
    priv_->texture_desc.mipmapFilterMode = cudaFilterModeLinear;
    priv_->texture_desc.disableTrilinearOptimization = 0;
    priv_->texture_desc.normalizedCoords = 0;
    priv_->texture_desc.readMode = cudaReadModeElementType;
}

CUQtTexture::CUQtTexture(const QImage &image, cudaStream_t stream):
      CUQtTexture()
{
    if (!image.isNull())
    {
        this->upload(image, stream);
    }
}

CUQtTexture::~CUQtTexture()
{
    if (priv_->handle)
    {
        cudaDestroyTextureObject(priv_->handle);
    }
}

bool CUQtTexture::isValid() const noexcept
{
    return handle() != 0 && this->priv_->memory && this->priv_->memory->isValid();
}

cudaTextureDesc CUQtTexture::textureDescriptor() const noexcept
{
    return priv_->texture_desc;
}

void CUQtTexture::setTextureDescriptor(const cudaTextureDesc &tex_desc) noexcept
{
    priv_->texture_desc = tex_desc;
}

cudaTextureObject_t CUQtTexture::handle() const noexcept
{
    return priv_->handle;
}

void *CUQtTexture::devicePointer() const noexcept
{
    return priv_->memory ? priv_->memory->devicePointer() : nullptr;
}

size_t CUQtTexture::pitch() const noexcept
{
    return priv_->memory ? priv_->memory->pitch() : 0;
}

cudaChannelFormatDesc CUQtTexture::formatDescriptor() const noexcept
{
    return priv_->format_desc;
}

cudaError CUQtTexture::preallocate(size_t width, size_t height, QImage::Format format, cudaStream_t stream)
{
    return cudaErrorNotYetImplemented;
}

cudaError CUQtTexture::upload(const QImage &image, cudaStream_t stream)
{
    if (image.isNull())
    {
        return cudaErrorInvalidHostPointer;
    }

    const auto format_desc{CUQt::toCudaChannelFormat(image.format())};
    if (format_desc.f == cudaChannelFormatKindNone)
    {
        return cudaErrorInvalidValue;
    }

    const auto width{image.width()};
    const auto height{image.height()};
    const auto bits_per_pixel{image.depth()};
    const auto width_in_bytes{(width * bits_per_pixel) / 8};

    //TODO add operator!= for cudaChannelFormatDesc
    //TODO change impl to modify priv_ only on success
    bool allocate_texture{!isValid() || std::memcmp(&priv_->format_desc, &format_desc, sizeof(format_desc))};
    if (width_in_bytes != priv_->allocated_width || height != priv_->allocated_height)
    {
        priv_->memory.reset(new CUQtDeviceMemoryBlock2D<uint8_t>(width_in_bytes, height));
        priv_->allocated_height = height;
        priv_->allocated_width = width_in_bytes;
        allocate_texture = true;
    }

    priv_->logical_width = width;
    this->priv_->format_desc = format_desc;
    const auto upload_result{this->priv_->memory->upload(image.bits(), image.bytesPerLine(), width_in_bytes, image.height())};
    if (upload_result != CUQt::MemcpyResult::MemcpySuccess)
    {
        return static_cast<cudaError>(upload_result);
    }

    priv_->allocated_format = image.format();
    if (allocate_texture)
    {
        if (priv_->handle)
        {
            cudaDestroyTextureObject(priv_->handle);
        }

        cudaResourceDesc resource_desc{};
        resource_desc.resType = cudaResourceTypePitch2D;
        resource_desc.res.pitch2D.desc = priv_->format_desc;
        resource_desc.res.pitch2D.devPtr = priv_->memory->devicePointer();
        resource_desc.res.pitch2D.height = priv_->allocated_height;
        resource_desc.res.pitch2D.pitchInBytes = priv_->memory->pitch();
        resource_desc.res.pitch2D.width = priv_->logical_width;
        return cudaCreateTextureObject(&priv_->handle, &resource_desc, &priv_->texture_desc, nullptr);
    }

    return cudaSuccess;
}

CUQtResult<QImage> CUQtTexture::download() const
{
    if (!isValid())
    {
        return {QImage(), cudaErrorInvalidValue};
    }
    QImage result{priv_->logical_width, priv_->allocated_height, priv_->allocated_format};
    const auto status{download(result)};
    return {result, status};
}

cudaError CUQtTexture::download(QImage &target) const
{
    if (!isValid())
    {
        return cudaErrorInvalidValue;
    }
    if (target.height() != priv_->allocated_height || target.width() != priv_->logical_width)
    {
        return cudaErrorInvalidConfiguration;
    }
    const auto status{priv_->memory->download(target.bits(), target.bytesPerLine(), priv_->allocated_width, target.height())};
    return static_cast<cudaError>(status);
}
