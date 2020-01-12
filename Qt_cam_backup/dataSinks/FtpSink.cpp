#include "FtpSink.h"

#include <QDateTime>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QQueue>
#include <QUrl>

namespace {
struct FtpUploadWorkItem {
    QByteArray source;
    QUrl target;
};
}

class FtpSink::Priv {
public:
    const FtpTarget settings;
    uint64_t fileCounter { 0 };
    QQueue<FtpUploadWorkItem> jobQueue;
    QNetworkAccessManager* nam { nullptr };
    bool uploadActive { false };
    std::function<void(const QString&)> statusMsg;

    explicit Priv(const FtpTarget& set) noexcept
        : settings { set }
    {
    }

    QString ftpBaseUrl() const
    {
        auto result = QString("ftp://%1@%2").arg(settings.user).arg(settings.host);
        if (!result.endsWith('/')) {
            result += '/';
        }
        return result;
    }
    QUrl prepareUrl(const QString& fileFmt) const
    {
        QUrl url(ftpBaseUrl() + currentDate() + '_' + QString::number(fileCounter) + '.' + fileFmt);
        url.setPort(21);
        url.setPassword(settings.passwd);
        return url;
    }

    void startUpload(const FtpUploadWorkItem& item)
    {
        statusMsg("start upload: " + item.target.path());
        QNetworkRequest req = QNetworkRequest(item.target);
        uploadActive = true;
        auto reply = nam->put(req, item.source);
        QObject::connect(reply, &QNetworkReply::finished, [this]() {
            qDebug() << "Upload finished";
            if (!jobQueue.empty()) {
                auto ftpUp = jobQueue.dequeue();
                statusMsg("pending uploads: " + QString::number(jobQueue.size()));
                startUpload(ftpUp);
            } else {
                statusMsg("Data uploaded to FTP");
                uploadActive = false;
            }
        });

        QObject::connect(reply, qOverload<QNetworkReply::NetworkError>(&QNetworkReply::error), [this, reply](auto err) {
            qDebug() << "Upload error: " << err;
            if (err != QNetworkReply::NoError) {
                statusMsg("Upload error: " + reply->errorString());
            }
        });
    }

    static QString currentDate()
    {
        const auto now = QDateTime::currentDateTime();
        return now.toString("dd_MM_yy_HH_mm_ss");
    }
};

FtpSink::FtpSink(const FtpTarget& ftpSettings, QObject* parent)
    : AbstractDataSink(parent)
    , priv_ { new Priv(ftpSettings) }
{
    priv_->nam = new QNetworkAccessManager(this);
    priv_->statusMsg = [this](const QString& message) {
        emit statusMessage(message);
    };
    QObject::connect(priv_->nam, &QNetworkAccessManager::finished, [this](QNetworkReply* reply) {
        qDebug() << reply->error();
        if (reply->error() != QNetworkReply::NoError) {
            emit statusMessage(reply->errorString());
        }
        reply->deleteLater();
    });
}

FtpSink::~FtpSink() = default;

bool FtpSink::isDone() const noexcept
{
    return !priv_->uploadActive && priv_->jobQueue.empty();
}

void FtpSink::process(AbstractDataSource* source, const QByteArray& data)
{
    //TODO: create folder for each upload
    ++priv_->fileCounter;
    FtpUploadWorkItem uploadItem { data, priv_->prepareUrl(source->preferredFileFormat()) };
    if (!priv_->uploadActive) {
        priv_->startUpload(uploadItem);
    } else {
        priv_->jobQueue.enqueue(std::move(uploadItem));
        emit statusMessage("pending uploads: " + QString::number(priv_->jobQueue.size()));
    }
}
