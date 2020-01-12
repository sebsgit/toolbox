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
    virtual QString preferredFileFormat() const = 0;
    // @brief returns true if the different data chunks can be merged into a single file
    virtual bool canMergeData() const noexcept = 0;
    virtual QByteArray dataSeparator() const noexcept { return QByteArray(); }

public slots:
    virtual void start() = 0;
    virtual void stop() = 0;

signals:
    void dataAvailable(const QByteArray& data);
    void statusMessage(const QString&);

public slots:
};

#endif // ABSTRACTDATASOURCE_H
