#include "gstdinlistener.h"

#include <iostream>
#include <QTimer>
#include <QFuture>
#include <QFutureWatcher>
#include <QtConcurrent/QtConcurrentRun>

class GStdInListener::Priv
{
public:
    QFutureWatcher<void> future_watcher;
};

GStdInListener::GStdInListener(QObject *parent) : QObject(parent),
    priv_{std::make_unique<Priv>()}
{
    QObject::connect(&priv_->future_watcher, &QFutureWatcher<void>::finished, this, &GStdInListener::readStdIn);
    QTimer::singleShot(0, this, &GStdInListener::readStdIn);
}

GStdInListener::~GStdInListener() = default;

void GStdInListener::readStdIn()
{
    auto future = QtConcurrent::run([this]() {
        std::string buffer;
        std::getline(std::cin, buffer);
        if (!buffer.empty()) {
            emit dataAvailable(QString::fromStdString(buffer));
        }
    });
    priv_->future_watcher.setFuture(future);

}
