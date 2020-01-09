#include "DataSinks.h"
#include "AbstractDataSink.h"
#include "dataSinks/DebugSink.h"
#include "dataSinks/FtpSink.h"

#include <QMetaMethod>
#include <QThread>
#include <vector>

#include <QDebug>

class DataSinks::Priv {
public:
    std::vector<std::unique_ptr<AbstractDataSink>> sinks;
    std::vector<QThread*> threads;
    QMetaMethod processMethod;
};

DataSinks::DataSinks(QObject* parent)
    : QObject(parent)
    , priv_(new Priv())
{
    const auto processMethodIndex = AbstractDataSink::staticMetaObject.indexOfMethod(QMetaObject::normalizedSignature("process(AbstractDataSource*, QByteArray)"));
    priv_->processMethod = AbstractDataSink::staticMetaObject.method(processMethodIndex);
    Q_ASSERT(priv_->processMethod.isValid());
}

DataSinks::~DataSinks()
{
    this->stop();
}

void DataSinks::configure(const BackupTargets& settings)
{
    this->stop();
    priv_->sinks.clear();
    priv_->sinks.push_back(std::make_unique<DebugSink>());
    if (settings.ftp.enabled) {
        priv_->sinks.push_back(std::make_unique<FtpSink>(settings.ftp));
    }
    for (auto& sink : priv_->sinks) {
        auto thread = std::make_unique<QThread>();
        sink->moveToThread(thread.get());
        thread->start();
        priv_->threads.push_back(thread.release());
    }
    emit configureDone();
}

void DataSinks::processData(AbstractDataSource* source, const QByteArray& data)
{
    for (auto& s : priv_->sinks)
        priv_->processMethod.invoke(s.get(), Qt::QueuedConnection, Q_ARG(AbstractDataSource*, source), Q_ARG(QByteArray, data));
}

void DataSinks::stop()
{
    for (auto thread : priv_->threads) {
        QObject::connect(thread, &QThread::finished, thread, &QObject::deleteLater);
        thread->quit();
    }
    priv_->threads.clear();
}
