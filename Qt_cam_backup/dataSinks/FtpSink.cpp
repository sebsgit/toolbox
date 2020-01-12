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
    bool canMerge { false };
    AbstractDataSource* origin { nullptr };
};
}

class FtpSink::Priv {
public:
    const FtpTarget settings;
    std::function<void(const QString&)> statusMsg;
    uint64_t fileCounter { 0 };
    QQueue<FtpUploadWorkItem> jobQueue;
    QNetworkAccessManager* nam { nullptr };
    const int32_t uploadLimitCount { 5 };
    int32_t currentActiveUploads { 0 };

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

    void enqueue(FtpUploadWorkItem&& uploadItem)
    {
        if (currentActiveUploads <= uploadLimitCount) {
            startUpload(uploadItem);
        } else {
            jobQueue.enqueue(std::move(uploadItem));
            if (jobQueue.size() > 10) {
                tryMergePendingData();
            }
        }
    }

private:
    void startUpload(const FtpUploadWorkItem& item)
    {
        statusMsg("start upload: " + item.target.path());
        QNetworkRequest req = QNetworkRequest(item.target);
        auto reply = nam->put(req, item.source);
        ++currentActiveUploads;
        QObject::connect(reply, &QNetworkReply::finished, [this]() {
            --currentActiveUploads;
            while (currentActiveUploads <= uploadLimitCount && !jobQueue.empty()) {
                if (!jobQueue.empty()) {
                    auto ftpUp = jobQueue.dequeue();
                    statusMsg("pending uploads: " + QString::number(jobQueue.size()) + ", active: " + QString::number(currentActiveUploads));
                    startUpload(ftpUp);
                }
            }
            if (jobQueue.empty() && currentActiveUploads == 0) {
                statusMsg("Data uploaded to FTP");
            }
        });

        QObject::connect(reply, qOverload<QNetworkReply::NetworkError>(&QNetworkReply::error), [this, reply](auto err) {
            qDebug() << "Upload error: " << err;
            if (err != QNetworkReply::NoError) {
                statusMsg("Upload error: " + reply->errorString());
            }
        });
    }

    void tryMergePendingData()
    {
        const int32_t countToMerge { std::count_if(jobQueue.begin(), jobQueue.end(), [](auto& item) { return item.canMerge; }) };
        if (countToMerge < 2) {
            return;
        }
        QByteArray mergedData;
        QUrl newTarget;
        AbstractDataSource* origin { nullptr };
        int32_t countMerged { 0 };
        for (int i = jobQueue.size() - 1; i >= 0; --i) {
            if (jobQueue[i].canMerge) {
                newTarget = jobQueue[i].target;
                if (origin) {
                    //TODO: separate by origin
                    Q_ASSERT(origin == jobQueue[i].origin);
                }
                origin = jobQueue[i].origin;
                mergedData += jobQueue[i].source + jobQueue[i].origin->dataSeparator();
                jobQueue.removeAt(i);
                ++countMerged;
            }
        }
        if (!mergedData.isEmpty()) {
            FtpUploadWorkItem item { std::move(mergedData), newTarget, true, origin };
            enqueue(std::move(item));
            statusMsg(QString("Merged %1 pending items into one upload").arg(countMerged));
        }
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
    return priv_->currentActiveUploads == 0 && priv_->jobQueue.empty();
}

void FtpSink::process(AbstractDataSource* source, const QByteArray& data)
{
    //TODO: create folder for each upload
    ++priv_->fileCounter;
    FtpUploadWorkItem uploadItem { data, priv_->prepareUrl(source->preferredFileFormat()), source->canMergeData(), source };
    priv_->enqueue(std::move(uploadItem));
}
