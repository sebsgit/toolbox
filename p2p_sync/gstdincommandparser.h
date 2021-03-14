#ifndef GSTDINCOMMANDPARSER_H
#define GSTDINCOMMANDPARSER_H

#include <QObject>

#include "gserver.h"

class GStdInCommandParser : public QObject
{
    Q_OBJECT
public:
    explicit GStdInCommandParser(GServer * server, QObject *parent = nullptr);

public slots:
    void executeCommand(const QString &cmd);

private:
    GServer *server_{nullptr};
};

#endif // GSTDINCOMMANDPARSER_H
