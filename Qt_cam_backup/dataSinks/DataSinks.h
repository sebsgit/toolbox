#ifndef DATASINKS_H
#define DATASINKS_H

#include <QObject>
#include <memory>

#include "AbstractDataSource.h"
#include "AppSettings.h"

class DataSinks : public QObject {
    Q_OBJECT
public:
    explicit DataSinks(QObject* parent = nullptr);
    ~DataSinks() override;

signals:
    void configureDone();
    void statusMessage(const QString&);

public slots:
    void configure(const BackupTargets& settings);
    void processData(AbstractDataSource* source, const QByteArray& data);
    void stop();
    void finalize();

private:
    class Priv;
    std::unique_ptr<Priv> priv_;
};

#endif // DATASINKS_H
