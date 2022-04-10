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
    int32_t allocated_width{0};
    int32_t allocated_height{0};
    int32_t logical_width{0};
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
    cudaError result{cudaSuccess};
    const auto format_desc{CUQt::toCudaChannelFormat(format)};
    if (format_desc.f != cudaChannelFormatKindNone)
    {
        const auto bits_per_pixel{format_desc.w + format_desc.x + format_desc.y + format_desc.z};
        const auto width_in_bytes{(width * bits_per_pixel) / 8};
        const bool allocate_memory{width_in_bytes != priv_->allocated_width || height != priv_->allocated_height};
        bool allocate_texture{!isValid() || priv_->format_desc != format_desc};
        if (allocate_memory || allocate_texture)
        {
            QScopedPointer<Priv> new_priv{new Priv()};
            new_priv->format_desc = priv_->format_desc;
            new_priv->texture_desc = priv_->texture_desc;
            new_priv->allocated_height = priv_->allocated_height;
            new_priv->allocated_width = priv_->allocated_width;
            new_priv->logical_width = priv_->logical_width;
            new_priv->allocated_format = priv_->allocated_format;

            if (allocate_memory)
            {
                new_priv->memory.reset(new CUQtDeviceMemoryBlock2D<uint8_t>(width_in_bytes, height));
                if (new_priv->memory->isValid())
                {
                    new_priv->logical_width = static_cast<int32_t>(width);
                    new_priv->allocated_height = static_cast<int32_t>(height);
                    new_priv->allocated_width = static_cast<int32_t>(width_in_bytes);
                    new_priv->format_desc = format_desc;
                    new_priv->allocated_format = format;
                    allocate_texture = true;
                }
                else
                {
                    result = cudaErrorMemoryAllocation;
                    allocate_texture = false;
                }
            }

            if (allocate_texture)
            {
                const auto device_memory_ptr{new_priv->memory ? new_priv->memory->devicePointer() : priv_->memory->devicePointer()};
                const auto memory_pitch{new_priv->memory ? new_priv->memory->pitch() : priv_->memory->pitch()};

                cudaResourceDesc resource_desc{};
                resource_desc.resType = cudaResourceTypePitch2D;
                resource_desc.res.pitch2D.desc = new_priv->format_desc;
                resource_desc.res.pitch2D.devPtr = device_memory_ptr;
                resource_desc.res.pitch2D.height = new_priv->allocated_height;
                resource_desc.res.pitch2D.pitchInBytes = memory_pitch;
                resource_desc.res.pitch2D.width = new_priv->logical_width;
                result = cudaCreateTextureObject(&new_priv->handle, &resource_desc, &priv_->texture_desc, nullptr);
            }

            if (result == cudaSuccess)
            {
                if (!new_priv->memory)
                {
                    new_priv->memory.swap(priv_->memory);
                }
                if (!new_priv->handle)
                {
                    new_priv->handle = priv_->handle;
                    priv_->handle = cudaTextureObject_t{};
                }
                if (priv_->handle)
                {
                    cudaDestroyTextureObject(priv_->handle);
                }
                priv_.swap(new_priv);
            }
        }
    }
    else
    {
        result = cudaErrorInvalidValue;
    }

    return result;
}

cudaError CUQtTexture::upload(const QImage &image, cudaStream_t stream)
{
    if (image.isNull())
    {
        return cudaErrorInvalidHostPointer;
    }

    const auto status{preallocate(image.width(), image.height(), image.format(), stream)};
    if (status != cudaSuccess)
    {
        return status;
    }

    const auto upload_result{this->priv_->memory->upload(image.bits(), image.bytesPerLine(), priv_->allocated_width, image.height())};
    if (upload_result != CUQt::MemcpyResult::MemcpySuccess)
    {
        return static_cast<cudaError>(upload_result);
    }

    return cudaSuccess;
}

CUQtResult<QImage> CUQtTexture::download() const
{
    if (!priv_->memory || priv_->logical_width == 0 || priv_->allocated_width == 0 || priv_->allocated_format == QImage::Format_Invalid)
    {
        return {QImage(), cudaErrorInvalidValue};
    }
    QImage result{priv_->logical_width, priv_->allocated_height, priv_->allocated_format};
    const auto status{download(result)};
    return {result, status};
}

cudaError CUQtTexture::download(QImage &target) const
{
    if (!priv_->memory)
    {
        return cudaErrorInvalidValue;
    }
    if (target.height() != priv_->allocated_height || target.width() != priv_->logical_width || target.format() != priv_->allocated_format)
    {
        return cudaErrorInvalidConfiguration;
    }
    const auto status{priv_->memory->download(target.bits(), target.bytesPerLine(), priv_->allocated_width, target.height())};
    return static_cast<cudaError>(status);
}
