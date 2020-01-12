#ifndef DEBUGSINK_H
#define DEBUGSINK_H

#include "AbstractDataSink.h"

class DebugSink : public AbstractDataSink {
    Q_OBJECT
public:
    explicit DebugSink(QObject* parent = nullptr);

    bool isDone() const noexcept override;

public slots:
    void process(AbstractDataSource* source, const QByteArray& data) override;
};

#endif // DEBUGSINK_H
