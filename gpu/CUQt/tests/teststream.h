#ifndef TESTSTREAM_H
#define TESTSTREAM_H

#include <QObject>
#include <qtestcase.h>

class TestStream : public QObject
{
    Q_OBJECT
public:
    explicit TestStream(QObject *parent = nullptr);

private slots:
    void defaultStreamStatus();
};

#endif // TESTSTREAM_H
