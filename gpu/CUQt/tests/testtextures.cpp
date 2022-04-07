#include "testtextures.h"

#include <cuqtgraphics.h>

TestTextures::TestTextures(QObject *parent)
    : QObject{parent}
{

}

void TestTextures::uploadQImage()
{
    QImage test_image{19, 15, QImage::Format_RGB32};
    for (auto y = 0; y < test_image.height(); ++y)
    {
        for (auto x = 0; x < test_image.width(); ++x)
        {
            test_image.setPixelColor(x, y, QColor(x * 10, y * 10, (x + y) * 5));
        }
    }

    CUQtTexture texture;
    QVERIFY(!texture.isValid());

    const auto status{texture.upload(test_image)};
    QCOMPARE(status, cudaSuccess);
    QVERIFY(texture.isValid());

    const auto download_result{texture.download()};
    QCOMPARE(download_result.status, cudaSuccess);
    QCOMPARE(download_result.value, test_image);
}
