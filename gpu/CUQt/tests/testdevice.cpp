#include <cuqtdevice.h>
#include <QDebug>
#include "testdevice.h"

TestDevice::TestDevice(QObject *parent)
    : QObject{parent}
{

}

void TestDevice::initTestCase()
{
    const auto runtime_version{CUQt::CUDAVersion()};
    const auto driver_version{CUQt::CUDADriverVersion()};
    qDebug() << "CUDA version:" << runtime_version;
    qDebug() << "Driver:" << driver_version;
}

void TestDevice::atLeastOneDeviceAvailable()
{
    QVERIFY(CUQtDevice::numberOfDevices() > 0);
}

void TestDevice::defaultDeviceIsValid()
{
    CUQtDevice device;
    QVERIFY(device.id() > -1);
    QVERIFY(!CUQt::hasError());
    QCOMPARE(CUQt::lastError(), cudaError::cudaSuccess);
    QVERIFY(!device.pciBusId().isEmpty());

    qDebug() << "Running tests on" << device.properties().name;
    qDebug() << "Compute caps:" << device.computeCapability();
}

void TestDevice::setInvalidId()
{
    CUQtDevice device;
    const auto id_before_set{device.id()};
    QVERIFY(id_before_set > -1);
    device.setId(-1);
    QVERIFY(CUQt::hasError());
    QCOMPARE(device.id(), id_before_set);
}
