#include "PictureDataSource.h"

#include <QBuffer>
#include <QCamera>
#include <QCameraImageCapture>
#include <QCameraInfo>
#include <QImageWriter>
#include <QTimer>
#include <QtGlobal>

#include <QDebug>

//TODO: processing queue
class PictureDataSource::Priv {
public:
    QCamera* camera = nullptr;
    QCameraInfo info;
    QCameraImageCapture* capture = nullptr;
    bool captureActive = true;
};

PictureDataSource::PictureDataSource(QObject* parent)
    : AbstractDataSource(parent)
    , priv_(new Priv())
{
    //TODO: select camera in settings
    for (const QCameraInfo& info : QCameraInfo::availableCameras()) {
        if (!info.isNull()) {
            priv_->info = info;
            priv_->camera = new QCamera(info, this);
            priv_->camera->setCaptureMode(QCamera::CaptureStillImage);
            priv_->capture = new QCameraImageCapture(priv_->camera, this);
            priv_->capture->setCaptureDestination(QCameraImageCapture::CaptureDestination::CaptureToBuffer);
            QObject::connect(priv_->capture, &QCameraImageCapture::imageCaptured, [this](int id, const QImage& image) {
                Q_UNUSED(id)
                QByteArray bytes;
                QBuffer buffer(&bytes);
                QImageWriter writer(&buffer, "JPEG");
                if (writer.write(image)) {
                    emit dataAvailable(bytes);
                }
            });
            break;
        }
    }
}

PictureDataSource::~PictureDataSource() = default;

QString PictureDataSource::name() const
{
    return priv_->info.description();
}

bool PictureDataSource::isActive() const
{
    return priv_->camera != nullptr;
}

QByteArray PictureDataSource::header() const
{
    return "data:pictures";
}

QString PictureDataSource::preferredFileFormat() const
{
    return "jpg";
}

bool PictureDataSource::canMergeData() const noexcept
{
    return false;
}

void PictureDataSource::start()
{
    priv_->captureActive = true;
    if (priv_->camera) {
        QObject::connect(priv_->camera, &QCamera::statusChanged, [this](QCamera::Status status) {
            if (status == QCamera::Status::ActiveStatus) {
                auto timer = new QTimer(this);
                //TODO: capture rate from settings
                timer->setInterval(1000);
                QObject::connect(timer, &QTimer::timeout, [this, timer]() {
                    if (priv_->captureActive) {
                        priv_->capture->capture();
                    } else {
                        timer->stop();
                    }
                });
                timer->start();
            }
        });
        priv_->camera->start();
    }
}

void PictureDataSource::stop()
{
    if (priv_->camera)
        priv_->camera->stop();
    priv_->captureActive = false;
}
