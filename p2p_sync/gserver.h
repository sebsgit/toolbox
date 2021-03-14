#ifndef GSERVER_H
#define GSERVER_H

#include <QObject>
#include <QHostAddress>
#include <memory>

#include "gcommandserializer.h"

class GServer : public QObject
{
    Q_OBJECT
public:
    explicit GServer(QObject *parent = nullptr);
    ~GServer() override;

public slots:
    void listen();
    void connectTo(const QHostAddress &address, const quint16 port);

public:
    void sendCommand(const GCommand &command);

signals:
    void criticalError(const QString &what);
    void infoMessage(const QString &what);

protected slots:
    void socketReadyRead();
    void socketError();

private:
    class Priv;
    std::unique_ptr<Priv> priv_;
};

#endif // GSERVER_H
