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
        QNetworkRequest req = QNetworkRequest(item.target);
        uploadActive = true;
        auto reply = nam->put(req, item.source);
        QObject::connect(reply, &QNetworkReply::finished, [this]() {
            qDebug() << "Upload finished";
            if (!jobQueue.empty()) {
                startUpload(jobQueue.dequeue());
            } else {
                uploadActive = false;
            }
        });

        QObject::connect(reply, qOverload<QNetworkReply::NetworkError>(&QNetworkReply::error), [](auto err) {
            qDebug() << "Upload error: " << err;
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
    QObject::connect(priv_->nam, &QNetworkAccessManager::finished, [](QNetworkReply* reply) {
        qDebug() << reply->error();
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
        qDebug() << "pending FTP uploads: " << priv_->jobQueue.size();
    }
}
