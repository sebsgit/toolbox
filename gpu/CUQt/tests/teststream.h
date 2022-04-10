#ifndef TESTSTREAM_H
#define TESTSTREAM_H

#include "testbase.h"

class TestStream : public QObject
{
    Q_OBJECT
public:
    explicit TestStream(QObject *parent = nullptr);

private slots:
    void defaultStreamStatus();
    void recordEventsInStream();
};

#endif // TESTSTREAM_H
