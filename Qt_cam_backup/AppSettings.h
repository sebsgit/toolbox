#ifndef APPSETTINGS_H
#define APPSETTINGS_H

#include <QObject>
#include <memory>

struct GpsSourceConfig {
    const bool enabled;
    const int updateIntervalMs;
};

struct DataSources {
    const GpsSourceConfig gps;
    const bool sound;
    const bool video;
    const bool pictures;
};

struct EmailTarget {
    const bool enabled;
    const QString path;
};

struct FtpTarget {
    const bool enabled;
    const QString host; //TODO: encrypted
    const QString user; //TODO: encrypted
    const QByteArray passwd; //TODO: encrypted
};

struct LocalStorage {
    const bool enabled;
    const QByteArray path; //TODO: encrypted
};

struct BackupTargets {
    const EmailTarget email;
    const FtpTarget ftp;
    const LocalStorage local;
};

struct SettingsData {
    const DataSources src;
    const BackupTargets backup;
};

class AppSettings : public QObject {
    Q_OBJECT
public:
    explicit AppSettings(QObject* parent = nullptr);
    ~AppSettings();

public:
    DataSources currentSourceSettings() const;
    BackupTargets currentTargetSettings() const;
    SettingsData currentSettings() const;

    bool isPINSaved() const;
    bool hasValidPINCode() const noexcept;
    bool isPINCodeValid(const QString& key) const;

public slots:
    void setPINCode(const QString& pin);

    void enableStillImageInput(bool enable);
    void enableSoundInput(bool enable);
    void enableGpsInput(bool enable);
    void enableVideoInput(bool enable);

    void enableLocalStorage(bool enable);
    void setLocalStoragePath(const QString& path);
    void enableBackupFtp(bool enable);
    void setFtpSettings(const FtpTarget& settings);

private:
    class Priv;
    std::unique_ptr<Priv> priv_;
};

#endif // APPSETTINGS_H
