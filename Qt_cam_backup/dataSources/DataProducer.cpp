#include "DataProducer.h"
#include "GpsDataSource.h"
#include "dataSources/PictureDataSource.h"

#include <vector>

class DataProducer::Priv {
public:
    std::vector<std::unique_ptr<AbstractDataSource>> sources;
};

DataProducer::DataProducer(QObject* parent)
    : QObject(parent)
    , priv_(new Priv())
{
}

DataProducer::~DataProducer() = default;

void DataProducer::configure(const DataSources& sourceSettings)
{
    priv_->sources.clear();
    if (sourceSettings.gps.enabled) {
        auto gps = std::make_unique<GpsDataSource>();
        if (gps->isActive()) {
            gps->setSampleInterval(sourceSettings.gps.updateIntervalMs);
            priv_->sources.push_back(std::move(gps));
        } else {
            emit error(tr("GPS data not available"));
        }
    }
    if (sourceSettings.pictures) {
        auto pictureSource = std::make_unique<PictureDataSource>();
        if (pictureSource->isActive()) {
            priv_->sources.push_back(std::move(pictureSource));
        } else {
            emit error(tr("Camera for still image capture not available"));
        }
    }
    for (auto& s : priv_->sources)
        QObject::connect(s.get(), &AbstractDataSource::dataAvailable, this, &DataProducer::onDataAvailable);
    emit configureDone();
}

void DataProducer::onDataAvailable(const QByteArray& data)
{
    Q_ASSERT(qobject_cast<AbstractDataSource*>(sender()));
    emit dataAvailable(static_cast<AbstractDataSource*>(sender()), data);
}

void DataProducer::start()
{
    for (auto& s : priv_->sources)
        s->start();
}

void DataProducer::stop()
{
    for (auto& s : priv_->sources)
        s->stop();
}
