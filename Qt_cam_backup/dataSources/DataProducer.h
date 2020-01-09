#ifndef DATAPRODUCER_H
#define DATAPRODUCER_H

#include "AbstractDataSource.h"
#include "AppSettings.h"
#include <QObject>

class DataProducer : public QObject {
    Q_OBJECT
public:
    explicit DataProducer(QObject* parent = nullptr);
    ~DataProducer() override;

signals:
    void error(const QString& desc);
    void configureDone();
    void dataAvailable(AbstractDataSource* source, const QByteArray& data);

public slots:
    void configure(const DataSources& sourceSettings);
    void start();
    void stop();

private slots:
    void onDataAvailable(const QByteArray& data);

private:
    class Priv;
    std::unique_ptr<Priv> priv_;
};

#endif // DATAPRODUCER_H
