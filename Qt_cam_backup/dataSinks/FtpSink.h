#ifndef FTPSINK_H
#define FTPSINK_H

#include "AbstractDataSink.h"
#include "DataStorageConfig.h"

#include <memory>

class FtpSink : public AbstractDataSink {
    Q_OBJECT
public:
    explicit FtpSink(const FtpTarget& ftpSettings, QObject* parent = nullptr);
    ~FtpSink() override;

    bool isDone() const noexcept override;

public slots:
    void process(AbstractDataSource* source, const QByteArray& data) override;

private:
    class Priv;
    std::unique_ptr<Priv> priv_;
};

#endif // FTPSINK_H
