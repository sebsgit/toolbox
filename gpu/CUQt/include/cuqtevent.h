#ifndef CUQTEVENT_H
#define CUQTEVENT_H

#include <QObject>
#include <QScopedPointer>
#include <chrono>
#include "cuqt.h"

class CUQT_DLL_SPECS CUQtEvent : public QObject
{
    Q_OBJECT
public:
    enum CreationFlag : uint8_t
    {
        Default = cudaEventDefault,
        BlockingSync = cudaEventBlockingSync,
        DisableTiming = cudaEventDisableTiming,
        Interprocess = cudaEventInterprocess
    };
    Q_DECLARE_FLAGS(CreationFlags, CreationFlag);
    Q_FLAG(CreationFlags);

    enum CompletionStatus : uint16_t
    {
        CompletionStatusDone = cudaSuccess,
        CompletionStatusNotReady = cudaErrorNotReady,
        CompletionStatusInvalidValue = cudaErrorInvalidValue,
        CompletionStatusInvalidHandle = cudaErrorInvalidResourceHandle,
        CompletionStatusLaunchFailure = cudaErrorLaunchFailure
    };
    Q_ENUM(CompletionStatus);

    enum RecordStatus : uint16_t
    {
        RecordStatusSuccess = cudaSuccess,
        RecordStatusInvalidValue = cudaErrorInvalidValue,
        RecordStatusInvalidHandle = cudaErrorInvalidResourceHandle,
        RecordStatusLaunchFailure = cudaErrorLaunchFailure
    };
    Q_ENUM(RecordStatus);

    explicit CUQtEvent(CreationFlags flags = CreationFlag::Default, QObject *parent = nullptr);
    ~CUQtEvent();

    operator cudaEvent_t() const noexcept;

    std::chrono::milliseconds elapsedTime(const cudaEvent_t to) const noexcept;

    CompletionStatus status() const noexcept;

public slots:
    void synchronize();
    CUQtEvent::RecordStatus record(cudaStream_t stream = 0);

private:
    class Priv;
    QScopedPointer<Priv> priv_;
};

#endif // CUQTEVENT_H
