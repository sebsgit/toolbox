#ifndef TESTDEVICEMEMORY_H
#define TESTDEVICEMEMORY_H

#include <QObject>
#include <qtestcase.h>

class TestDeviceMemory : public QObject
{
Q_OBJECT
public:
    explicit TestDeviceMemory(QObject *parent = nullptr);

private slots:
    void uploadAndDownload();
};

#endif // TESTDEVICEMEMORY_H
