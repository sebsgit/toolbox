#include "testdevicememory.h"
#include <cuqtdevice.h>
#include <cuqtmemory.h>
#include <vector>

#include <QDebug>

__global__ void addition_kernel(float *a, float *b, float *c)
{
    const auto idx{blockIdx.x * blockDim.x + threadIdx.x};
    c[idx] = a[idx] + b[idx];
}

TestDeviceMemory::TestDeviceMemory(QObject *parent)
    : QObject{parent}
{

}

void TestDeviceMemory::uploadAndDownload()
{
    CUQtDevice default_device;

    float a{10.0f}, b{20.0f}, c{0.0f};

    CUQtDeviceMemoryBlock<float> dev_a{1};
    CUQtDeviceMemoryBlock<float> dev_b{1};
    CUQtDeviceMemoryBlock<float> dev_c{1};

    QVERIFY(dev_a.isValid());
    QVERIFY(dev_b.isValid());
    QVERIFY(dev_c.isValid());

    auto res = dev_a.upload(&a, 1);
    QCOMPARE(res, CUQt::MemcpySuccess);

    res = dev_b.upload(&b, 1);
    QCOMPARE(res, CUQt::MemcpySuccess);

    addition_kernel<<<dim3{1}, dim3{1}>>>(dev_a.devicePointer(), dev_b.devicePointer(), dev_c.devicePointer());
    default_device.synchronize();

    res = dev_c.download(&c, 1);
    QCOMPARE(res, CUQt::MemcpySuccess);
    QCOMPARE(c, 30.0f);
}
