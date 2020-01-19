#include "GpsDataSource.h"

#include <QDebug>
#include <QGeoPositionInfoSource>
#include <QTimer>

class GpsDataSource::Priv {
public:
    QGeoPositionInfoSource* source = nullptr;
    QDateTime lastUpdate;
    QTimer watchdog;
};

GpsDataSource::GpsDataSource(QObject* parent)
    : AbstractDataSource(parent)
    , priv_(new Priv())
{
    priv_->source = QGeoPositionInfoSource::createDefaultSource(this);
    if (priv_->source) {
        QObject::connect(priv_->source, &QGeoPositionInfoSource::positionUpdated, this, &GpsDataSource::gpsDataAvailable);
        QObject::connect(priv_->source, qOverload<QGeoPositionInfoSource::Error>(&QGeoPositionInfoSource::error), [this](auto err) {
            qDebug() << "GPS source error: " << err;
            emit statusMessage("GPS error: " + QString::number(err));
        });
        QObject::connect(&priv_->watchdog, &QTimer::timeout, [this]() {
            if (priv_->source->error() == QGeoPositionInfoSource::Error::NoError) {
                if (priv_->lastUpdate.msecsTo(QDateTime::currentDateTime()) > priv_->source->updateInterval() * 2) {
                    auto data = priv_->source->lastKnownPosition();
                    if (data.isValid()) {
                        qDebug() << "Watchdog update: " << data.coordinate().toString();
                        priv_->lastUpdate = QDateTime::currentDateTime();
                        emit dataAvailable(data.coordinate().toString().toLatin1());
                    }
                }
            }
        });
    } else {
        qDebug() << "GPS device not available";
    }
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

QString GpsDataSource::preferredFileFormat() const
{
    return "gps";
}

bool GpsDataSource::canMergeData() const noexcept
{
    return true;
}

QByteArray GpsDataSource::dataSeparator() const noexcept
{
    return QByteArray("\n");
}

void GpsDataSource::setSampleInterval(int msec)
{
    if (priv_->source)
        priv_->source->setUpdateInterval(msec);
}

void GpsDataSource::start()
{
    if (priv_->source) {
        priv_->watchdog.start(priv_->source->updateInterval() * 2);
        priv_->source->startUpdates();
    }
}

void GpsDataSource::stop()
{
    priv_->watchdog.stop();
    if (priv_->source)
        priv_->source->stopUpdates();
}

void GpsDataSource::gpsDataAvailable(const QGeoPositionInfo& info)
{
    qDebug() << info.coordinate().toString();
    if (info.isValid()) {
        QByteArray data = info.timestamp().toString().toLatin1() + " | " + info.coordinate().toString().toLatin1();
        emit dataAvailable(data);
        priv_->lastUpdate = QDateTime::currentDateTime();
    }
}
