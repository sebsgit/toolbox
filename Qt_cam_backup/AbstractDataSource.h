#ifndef ABSTRACTDATASOURCE_H
#define ABSTRACTDATASOURCE_H

#include <QByteArray>
#include <QObject>

class AbstractDataSource : public QObject {
    Q_OBJECT
public:
    explicit AbstractDataSource(QObject* parent = nullptr);

    virtual QString name() const = 0;
    virtual bool isActive() const = 0;
    virtual QByteArray header() const = 0;

public slots:
    virtual void start() = 0;
    virtual void stop() = 0;

signals:
    void dataAvailable(const QByteArray& data);

public slots:
};

#endif // ABSTRACTDATASOURCE_H
