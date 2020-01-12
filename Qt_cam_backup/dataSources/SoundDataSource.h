#ifndef SOUNDDATASOURCE_H
#define SOUNDDATASOURCE_H

#include "AbstractDataSource.h"

#include <memory>

class SoundDataSource : public AbstractDataSource {
    Q_OBJECT
public:
    explicit SoundDataSource(QObject* parent = nullptr);
    ~SoundDataSource() override;

    QString name() const override;
    bool isActive() const override;
    QByteArray header() const override;
    QString preferredFileFormat() const override;
    bool canMergeData() const noexcept override;

public slots:
    void start() override;
    void stop() override;

private:
    class Priv;
    std::unique_ptr<Priv> priv_;
};

#endif // SOUNDDATASOURCE_H
