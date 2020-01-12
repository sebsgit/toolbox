#ifndef PICTUREDATASOURCE_H
#define PICTUREDATASOURCE_H

#include "AbstractDataSource.h"
#include <memory>

class PictureDataSource : public AbstractDataSource {
    Q_OBJECT
public:
    explicit PictureDataSource(QObject* parent = nullptr);
    ~PictureDataSource() override;

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

#endif // PICTUREDATASOURCE_H
