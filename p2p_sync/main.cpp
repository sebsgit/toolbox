#include <QCoreApplication>
#include <QDebug>

#include "gserver.h"
#include "gcommandserializer.h"
#include "gstdinlistener.h"
#include "gstdincommandparser.h"

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    const auto args{a.arguments()};

    qDebug() << "Using data serializer " << GCommandSerializer::version();

    GServer server;
    QObject::connect(&server, &GServer::criticalError, &server, [](auto err)
     {
        qDebug() << err;
     });
    QObject::connect(&server, &GServer::infoMessage, &server, [](auto info)
     {
         qDebug() << info;
     });

    if (args.contains("--connect"))
    {
        auto pos = args.indexOf("--connect");
        if (pos < args.length() - 2)
        {
            const auto address{args[pos + 1]};
            const auto port{args[pos + 2].toUInt()};
            server.connectTo(QHostAddress(address), port);
        }
    }
    else
    {
        server.listen();
    }

    GStdInCommandParser command_parser{&server};
    GStdInListener stdin_socket;
    QObject::connect(&stdin_socket, &GStdInListener::dataAvailable,
                     &command_parser, &GStdInCommandParser::executeCommand,
                     Qt::QueuedConnection);
    return a.exec();
}
