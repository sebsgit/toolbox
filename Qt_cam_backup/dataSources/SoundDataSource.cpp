#include "SoundDataSource.h"

#include <QAudioBuffer>
#include <QAudioEncoderSettings>
#include <QAudioProbe>
#include <QAudioRecorder>
#include <QFile>
#include <QTimer>
#include <QUrl>

class SoundDataSource::Priv {
public:
    QAudioRecorder recorder;
    QUrl outputLocation;
    QTimer timer;
    bool recordNext { true };
    int32_t updateIntervalMs { 3 * 1000 }; //TODO: from settings
};

SoundDataSource::SoundDataSource(QObject* parent)
    : AbstractDataSource(parent)
    , priv_ { std::make_unique<Priv>() }
{
    qDebug() << priv_->recorder.audioInputs();
    if (!priv_->recorder.audioInputs().empty()) {
        QAudioEncoderSettings settings;
        settings.setQuality(QMultimedia::EncodingQuality::NormalQuality); //TODO: from settings
        priv_->recorder.setEncodingSettings(settings);

        QObject::connect(&priv_->recorder, qOverload<QMediaRecorder::Error>(&QMediaRecorder::error), [](auto err) {
            qDebug() << "recorder error: " << err;
        });
        QObject::connect(&priv_->recorder, &QMediaRecorder::statusChanged, [this](auto status) {
            qDebug() << "recorder status: " << status;
            if (status == QMediaRecorder::LoadedStatus) {
                if (priv_->recordNext) {
                    qDebug() << "record next";
                    priv_->recorder.record();
                }
            }
        });
        QObject::connect(&priv_->recorder, &QMediaRecorder::stateChanged, [this](auto state) {
            qDebug() << "recorder state: " << state;
            if (state == QMediaRecorder::State::StoppedState && priv_->outputLocation.isValid()) {
                QFile file(priv_->outputLocation.toLocalFile());
                if (file.open(QIODevice::ReadOnly)) {
                    emit dataAvailable(file.readAll());
                }
                priv_->outputLocation.clear();
            }
            if (state == QMediaRecorder::State::RecordingState) {
                QTimer::singleShot(priv_->updateIntervalMs, &priv_->recorder, &QMediaRecorder::stop);
            }
        });
        QObject::connect(&priv_->recorder, &QMediaRecorder::actualLocationChanged, [this](auto location) {
            priv_->outputLocation = location;
            qDebug() << "output loc: " << location;
        });
        auto probe = new QAudioProbe(this);
        const bool status = probe->setSource(&priv_->recorder);
        if (status) {
            QObject::connect(probe, &QAudioProbe::audioBufferProbed, [](const QAudioBuffer& buffer) {
                qDebug() << "Received audio buffer: " << buffer.format() << buffer.byteCount();
            });
        } else {
            qDebug() << "Can't set the audio probe";
        }
    }
}

SoundDataSource::~SoundDataSource() = default;

QString SoundDataSource::name() const
{
    return "SoundDevice: " + priv_->recorder.audioInput();
}

bool SoundDataSource::isActive() const
{
    return !priv_->recorder.audioInputs().empty();
}

QByteArray SoundDataSource::header() const
{
    return QByteArray("data:sound");
}

QString SoundDataSource::preferredFileFormat() const
{
    return "mp4";
}

bool SoundDataSource::canMergeData() const noexcept
{
    return false;
}

void SoundDataSource::start()
{
    priv_->recorder.record();
}

void SoundDataSource::stop()
{
    priv_->recorder.stop();
    priv_->recordNext = false;
}
