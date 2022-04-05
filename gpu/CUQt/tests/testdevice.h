#ifndef TESTDEVICE_H
#define TESTDEVICE_H

#include "testbase.h"

class TestDevice : public QObject
{
    Q_OBJECT
public:
    explicit TestDevice(QObject *parent = nullptr);

private slots:
    void initTestCase();
    void atLeastOneDeviceAvailable();
    void defaultDeviceIsValid();
    void setInvalidId();

};

#endif // TESTDEVICE_H
