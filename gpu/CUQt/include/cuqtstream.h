#ifndef CUQTSTREAM_H
#define CUQTSTREAM_H

#include <QObject>
#include <QScopedPointer>

#include "cuqt.h"

class CUQT_DLL_SPECS CUQtStream : public QObject
{
    Q_OBJECT
public:
    enum CreationFlags : int8_t
    {
        Default = cudaStreamDefault,
        NonBlocking = cudaStreamNonBlocking
    };
    Q_ENUM(CreationFlags);

    enum CompletionStatus : int8_t
    {
        Done,
        NotReady,
        Unknown
    };
    Q_ENUM(CompletionStatus);

    explicit CUQtStream(CreationFlags flags = CreationFlags::Default, QObject *parent = nullptr);
    ~CUQtStream();

    CompletionStatus status() const noexcept;

public slots:
    void synchronize();

signals:
    void completed();

private:
    class Priv;
    QScopedPointer<Priv> priv_;
};

#endif // CUQTSTREAM_H
