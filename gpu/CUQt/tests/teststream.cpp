#include <cuqtdevice.h>
#include <cuqtstream.h>
#include <cuqtmemory.h>
#include <cuqtevent.h>
#include "teststream.h"

#include <QElapsedTimer>

TestStream::TestStream(QObject *parent)
    : QObject{parent}
{

}

void TestStream::defaultStreamStatus()
{
    CUQtStream stream;
    QVERIFY(!CUQt::hasError());
    QCOMPARE(stream.status(), CUQtStream::CompletionStatus::Done);
}

__global__ void do_some_work(float *out, int n_per_thread)
{
    const auto id{blockIdx.x * blockDim.x + threadIdx.x};
    for (auto i = id * n_per_thread; i < (id + 1) * n_per_thread; ++i)
    {
        out[i] = sqrt(powf(i % 13, 3.14 + 1.0 / (id + 1.0)));
    }
}

void TestStream::recordEventsInStream()
{
    const size_t elements_per_thread{32 * 2048};
    const size_t n_threads{32};
    const size_t n_blocks{32};
    const size_t number_of_elements{n_blocks * n_threads * elements_per_thread};

    std::vector<float> result_data;
    result_data.resize(number_of_elements, 0.0f);

    CUQtStream stream{CUQtStream::CreationFlags::NonBlocking};
    CUQtDeviceMemoryBlock<float> data{number_of_elements};
    QCOMPARE(CUQt::lastError(), cudaSuccess);

    CUQtEvent start_event;
    CUQtEvent kernel_done_event;
    CUQtEvent memcpy_done_event;

    auto event_status{start_event.record(stream)};
    QCOMPARE(event_status, CUQtEvent::RecordStatus::RecordStatusSuccess);

    QElapsedTimer timer;
    timer.start();

    do_some_work <<< dim3(n_blocks), dim3(n_threads), 0, stream >>> (data.devicePointer(), elements_per_thread);
    event_status = kernel_done_event.record(stream);
    QCOMPARE(event_status, CUQtEvent::RecordStatus::RecordStatusSuccess);

    data.download(result_data.data(), number_of_elements, stream);
    event_status = memcpy_done_event.record(stream);
    QCOMPARE(event_status, CUQtEvent::RecordStatus::RecordStatusSuccess);
    memcpy_done_event.synchronize();

    const auto host_ms_after_sync{timer.elapsed()};

    QVERIFY(start_event.elapsedTime(memcpy_done_event) > start_event.elapsedTime(kernel_done_event));
    QVERIFY(host_ms_after_sync > start_event.elapsedTime(memcpy_done_event).count());
}
