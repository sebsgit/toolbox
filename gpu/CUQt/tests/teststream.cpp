#include <cuqtdevice.h>
#include <cuqtstream.h>
#include "teststream.h"

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
