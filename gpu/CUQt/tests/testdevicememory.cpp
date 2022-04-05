#include <vector>
#include <cuqtdevice.h>
#include <cuqtmemory.h>
#include "testdevicememory.h"

__global__ void addition_kernel(float *a, float *b, float *c)
{
    const auto idx{blockIdx.x * blockDim.x + threadIdx.x};
    c[idx] = a[idx] + b[idx];
}

__global__ void kernel_2d(const int32_t * src, size_t pitch_src, int32_t *out, size_t pitch_dest)
{
    const auto idx{blockIdx.x * blockDim.x + threadIdx.x};
    out[threadIdx.x + threadIdx.y * pitch_dest / sizeof(*out)] = src[threadIdx.x + threadIdx.y * pitch_src / sizeof(*src)] * 2;
}

TestDeviceMemory::TestDeviceMemory(QObject *parent)
    : QObject{parent}
{

}

void TestDeviceMemory::uploadAndDownload1D()
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

void TestDeviceMemory::uploadAndDownload2D()
{
    CUQtDevice default_device;

    const size_t width{13};
    const size_t height{47};

    CUQtDeviceMemoryBlock2D<int32_t> mat1{width, height};
    CUQtDeviceMemoryBlock2D<int32_t> mat2{width, height};
    QVERIFY(mat1.isValid());
    QVERIFY(mat2.isValid());

    QVERIFY(mat1.pitch() >= width * sizeof(int32_t));
    QCOMPARE(mat1.pitch(), mat2.pitch());

    std::vector<int32_t> host_memory(width * height);
    std::vector<int32_t> result(width * height);

    std::iota(host_memory.begin(), host_memory.end(), 0);
    std::fill(result.begin(), result.end(), 0);

    QCOMPARE(host_memory.size(), width * height);
    QCOMPARE(result.size(), width * height);

    for (size_t i = 1; i < result.size(); ++i)
    {
        QVERIFY(result[i] != host_memory[i] * 2);
    }

    auto res = mat1.upload(host_memory.data(), width * sizeof(int32_t), width, height);
    QCOMPARE(res, CUQt::MemcpySuccess);

    kernel_2d<<<dim3{1}, dim3{width, height}>>>(mat1.devicePointer(), mat1.pitch(), mat2.devicePointer(), mat2.pitch());
    default_device.synchronize();

    QCOMPARE(CUQt::lastError(), cudaSuccess);

    res = mat2.download(result.data(), width * sizeof(int32_t), width, height);
    QCOMPARE(res, CUQt::MemcpySuccess);

    for (size_t i = 0; i < result.size(); ++i)
    {
        QCOMPARE(result[i], host_memory[i] * 2);
    }
}
