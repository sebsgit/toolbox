#ifndef TESTDEVICEMEMORY_H
#define TESTDEVICEMEMORY_H

#include "testbase.h"

class TestDeviceMemory : public QObject
{
Q_OBJECT
public:
    explicit TestDeviceMemory(QObject *parent = nullptr);

private slots:
    void uploadAndDownload1D();
    void uploadAndDownload2D();
};

#endif // TESTDEVICEMEMORY_H
