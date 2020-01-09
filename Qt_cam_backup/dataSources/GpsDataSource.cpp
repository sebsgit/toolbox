#include "GpsDataSource.h"

#include <QGeoPositionInfoSource>

class GpsDataSource::Priv {
public:
    QGeoPositionInfoSource* source = nullptr;
};

GpsDataSource::GpsDataSource(QObject* parent)
    : AbstractDataSource(parent)
    , priv_(new Priv())
{
    priv_->source = QGeoPositionInfoSource::createDefaultSource(this);
    if (priv_->source)
        QObject::connect(priv_->source, &QGeoPositionInfoSource::positionUpdated, this, &GpsDataSource::gpsDataAvailable);
}

GpsDataSource::~GpsDataSource() = default;

QString GpsDataSource::name() const
{
    return "GPS";
}

bool GpsDataSource::isActive() const
{
    return priv_->source != nullptr;
}

QByteArray GpsDataSource::header() const
{
    return QByteArray("data:gps");
}

void GpsDataSource::setSampleInterval(int msec)
{
    if (priv_->source)
        priv_->source->setUpdateInterval(msec);
}

void GpsDataSource::start()
{
    if (priv_->source)
        priv_->source->startUpdates();
}

void GpsDataSource::stop()
{
    if (priv_->source)
        priv_->source->stopUpdates();
}

void GpsDataSource::gpsDataAvailable(const QGeoPositionInfo& info)
{
    QByteArray data;
    //TODO
    data = "fill me with gps data";
    emit dataAvailable(data);
}
