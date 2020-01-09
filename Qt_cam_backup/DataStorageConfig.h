#ifndef DATASTORAGECONFIG_H
#define DATASTORAGECONFIG_H

#include <QWidget>
#include <memory>

#include "AppSettings.h"

class DataStorageConfig : public QWidget {
    Q_OBJECT

public:
    explicit DataStorageConfig(AppSettings& settings, QWidget* parent = nullptr);
    ~DataStorageConfig();

private slots:
    void onGroupBoxLocalChecked(bool checked);
    void selectLocalStoragePath();

private:
    FtpTarget ftpSettings() const noexcept;

private:
    class Priv;
    std::unique_ptr<Priv> priv_;
};

#endif // DATASTORAGECONFIG_H
