#ifndef ABSTRACTDATASINK_H
#define ABSTRACTDATASINK_H

#include <QObject>

#include "AbstractDataSource.h"

class AbstractDataSink : public QObject {
    Q_OBJECT
public:
    explicit AbstractDataSink(QObject* parent = nullptr);

signals:

public slots:
    virtual void process(AbstractDataSource* source, const QByteArray& data) = 0;
};

#endif // ABSTRACTDATASINK_H
