#ifndef GPSDATASOURCE_H
#define GPSDATASOURCE_H

#include "AbstractDataSource.h"
#include <QGeoPositionInfo>
#include <memory>

class GpsDataSource : public AbstractDataSource {
    Q_OBJECT
public:
    GpsDataSource(QObject* parent = nullptr);
    ~GpsDataSource() override;

    QString name() const override;
    bool isActive() const override;
    QByteArray header() const override;
    QString preferredFileFormat() const override;

public slots:
    void setSampleInterval(int msec);
    void start() override;
    void stop() override;

private slots:
    void gpsDataAvailable(const QGeoPositionInfo& info);

private:
    class Priv;
    std::unique_ptr<Priv> priv_;
};

#endif // GPSDATASOURCE_H
