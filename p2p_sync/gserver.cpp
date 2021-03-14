#include "gserver.h"

#include <QTcpServer>
#include <QTcpSocket>
#include <QDebug>
#include <QNetworkProxy>

class GServer::Priv
{
public:
    QTcpServer server;
    QTcpSocket *connected_client{nullptr};

    GCommandSerializer serializer;
};

GServer::GServer(QObject *parent) : QObject(parent),
    priv_{std::make_unique<Priv>()}
{
    priv_->server.setProxy(QNetworkProxy::NoProxy);
    QObject::connect(&priv_->server, &QTcpServer::newConnection, this, [this]() {
        if (priv_->connected_client)
        {
            emit criticalError("Mutliple connections are not supported");
            return;
        }
        priv_->connected_client = priv_->server.nextPendingConnection();
        QObject::connect(priv_->connected_client, &QTcpSocket::readyRead,
                         this, &GServer::socketReadyRead);
        QObject::connect(priv_->connected_client, SIGNAL(error(QAbstractSocket::SocketError)), this, SLOT(socketError()));
    });

    QObject::connect(&priv_->server, &QTcpServer::acceptError, this, [this]() {
        emit criticalError("Server accept error: " + priv_->server.errorString());
    });
}

GServer::~GServer()
{
    if (priv_->connected_client)
    {
        priv_->connected_client->close();
    }
}

void GServer::socketError()
{
   emit criticalError("Socket error: " + priv_->connected_client->errorString());
   priv_->connected_client->deleteLater();
   priv_->connected_client = nullptr;
}

void GServer::listen()
{
    if (priv_->server.isListening())
    {
        return;
    }

    if (priv_->connected_client)
    {
        emit infoMessage("The client is already connected");
        return;
    }

    if (!priv_->server.listen(QHostAddress::AnyIPv4))
    {
        emit criticalError("Server listen failed: " + priv_->server.errorString());
    }
    else
    {
        qDebug() << "Server listen on " << priv_->server.serverAddress() << priv_->server.serverPort();
    }
}

void GServer::connectTo(const QHostAddress &address, quint16 port)
{
    if (priv_->connected_client)
    {
        emit infoMessage("The client is already connected");
        return;
    }

    priv_->connected_client = new QTcpSocket(this);
    priv_->connected_client->setProxy(QNetworkProxy::NoProxy);
    QObject::connect(priv_->connected_client, SIGNAL(error(QAbstractSocket::SocketError)), this, SLOT(socketError()));
    QObject::connect(priv_->connected_client, &QAbstractSocket::hostFound, this, [this]() {
        emit infoMessage("Host found");
    });
    QObject::connect(priv_->connected_client, &QAbstractSocket::connected, this, [this]() {
        emit infoMessage("Connected");

        GConsoleOutput version_sync{"Client serializer version is " + GCommandSerializer::version().toHex()};
        sendCommand(version_sync);
    });
    QObject::connect(priv_->connected_client, &QAbstractSocket::readyRead,
                     this, &GServer::socketReadyRead);

    priv_->connected_client->connectToHost(address, port);
}

void GServer::socketReadyRead()
{
    if (!priv_->connected_client)
    {
        return;
    }

    priv_->serializer.addData(priv_->connected_client->readAll());

    while (auto cmd = priv_->serializer.pendingCommand())
    {
        QObject::connect(cmd, &GCommand::outputAvailable,
                         this, [this](const QByteArray &outp) {
                             GConsoleOutput print_cmd{outp};
                             sendCommand(print_cmd);
                         });
        qDebug() << "Executing remote command - " << cmd->description();
        QObject::connect(cmd, &GCommand::finished, cmd, &QObject::deleteLater);
        cmd->execute();
    }
}

void GServer::sendCommand(const GCommand &command)
{
    if (!priv_->connected_client)
    {
        qDebug() << "Client not connected";
        return;
    }

    const auto raw_data{command.serialize()};
    priv_->connected_client->write(raw_data.header.toByteArray());
    priv_->connected_client->write(raw_data.data);
}
