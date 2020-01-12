#include "DebugSink.h"
#include <QDebug>

DebugSink::DebugSink(QObject* parent)
    : AbstractDataSink(parent)
{
}

bool DebugSink::isDone() const noexcept
{
    return true;
}

void DebugSink::process(AbstractDataSource* source, const QByteArray& data)
{
    qDebug() << source->name() << ": " << data.size() << " bytes of data...";
}
