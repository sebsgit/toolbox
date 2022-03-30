#ifndef CUQTDEVICE_H
#define CUQTDEVICE_H

#include <QObject>
#include <QScopedPointer>

#include <cuda_runtime.h>

#include "cuqt.h"

class CUQT_DLL_SPECS CUQtDevice : public QObject
{
    Q_OBJECT
public:
    static size_t numberOfDevices();

    explicit CUQtDevice(QObject *parent = nullptr);
    ~CUQtDevice();

    int32_t id() const noexcept;

    cudaDeviceProp properties() const noexcept;
    QByteArray pciBusId() const noexcept;

public slots:
    void setId(const int32_t id) noexcept;
    void synchronize();

private:
    class Priv;
    QScopedPointer<Priv> priv_;
};

#endif // CUQTDEVICE_H
