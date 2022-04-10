#include "testtextures.h"

#include <cuqtdevice.h>
#include <cuqtgraphics.h>

static QImage generateTestImage(int w, int h)
{
    QImage test_image{w, h, QImage::Format_RGB32};
    for (auto y = 0; y < test_image.height(); ++y)
    {
        for (auto x = 0; x < test_image.width(); ++x)
        {
            test_image.setPixelColor(x, y, QColor((x * 10) % 255, (y * 5) % 255, ((x + y) * 6) % 255, 255));
        }
    }

    return test_image;
}

TestTextures::TestTextures(QObject *parent)
    : QObject{parent}
{

}

void TestTextures::uploadQImage()
{
    QImage test_image{generateTestImage(15, 19)};

    CUQtTexture texture;
    QVERIFY(!texture.isValid());

    const auto status{texture.upload(test_image)};
    QCOMPARE(status, cudaSuccess);
    QVERIFY(texture.isValid());

    const auto download_result{texture.download()};
    QCOMPARE(download_result.status, cudaSuccess);
    QCOMPARE(download_result.value, test_image);
}

__global__ static void copyTexture(cudaTextureObject_t tex, uchar4 *rgba_out, size_t output_pitch)
{
    const auto color{tex2D<uchar4>(tex, threadIdx.x, threadIdx.y)};
    const auto output_offset{threadIdx.y * output_pitch / 4 + threadIdx.x};
    rgba_out[output_offset] = color;
}

void TestTextures::readWriteTextureMemory()
{
    QImage test_image{generateTestImage(32, 32)};
    QImage output_image{test_image.size(), test_image.format()};
    output_image.fill(QColor(0, 0, 0));

    QVERIFY(test_image != output_image);

    CUQtTexture texture;
    auto status{texture.upload(test_image)};
    QCOMPARE(status, cudaSuccess);
    QVERIFY(texture.isValid());

    CUQtTexture output_texture;
    status = output_texture.preallocate(test_image.width(), test_image.height(), test_image.format());
    QCOMPARE(status, cudaSuccess);
    QVERIFY(output_texture.isValid());
    QVERIFY(output_texture.devicePointer() != nullptr);

    const dim3 grid_size{1};
    const dim3 block_size{static_cast<uint32_t>(test_image.width()), static_cast<uint32_t>(test_image.height()), 1};
    copyTexture<<<grid_size, block_size>>>(texture.handle(), static_cast<uchar4*>(output_texture.devicePointer()), output_texture.pitch());
    QCOMPARE(CUQt::lastError(), cudaSuccess);

    const auto result{output_texture.download()};
    QCOMPARE(result.status, cudaSuccess);
    QCOMPARE(result.value, test_image);
}
