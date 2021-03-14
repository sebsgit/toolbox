#ifndef GSTDINLISTENER_H
#define GSTDINLISTENER_H

#include <QObject>
#include <memory>

class GStdInListener : public QObject
{
    Q_OBJECT
public:
    explicit GStdInListener(QObject *parent = nullptr);
    ~GStdInListener();

signals:
    void dataAvailable(const QString &);

private slots:
    void readStdIn();

private:
    class Priv;
    std::unique_ptr<Priv> priv_;
};

#endif // GSTDINLISTENER_H
