#include "teststream.h"
#include <cuqtdevice.h>
#include <cuqtstream.h>

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
