#include "AppSettings.h"
#include <QSettings>

class AppSettings::Priv {
public:
    QSettings settings { "BackupAppDomain", "BackupApp" };

    constexpr static const char* const srcGpsKey = "src/gps";
    constexpr static const char* const srcGpsUpdateIntervalKey = "src/gps_update_interval";
    constexpr static const char* const srcSoundKey = "src/sound";
    constexpr static const char* const srcVideoKey = "src/video";
    constexpr static const char* const srcPicsKey = "src/pictures";

    constexpr static const char* const targetEmailEnabledKey = "target/email_enable";
    constexpr static const char* const targetFtpEnabledKey = "target/ftp_enable";
    constexpr static const char* const targetLocalEnabledKey = "target/local_enable";

    constexpr static const char* const targetEmailKey = "target/email";
    constexpr static const char* const targetFtpHostKey = "target/ftp_host";
    constexpr static const char* const targetFtpUserKey = "target/ftp_user";
    constexpr static const char* const targetFtpPasswdKey = "target/ftp_pass";
    constexpr static const char* const targetLocalPathKey = "target/local_path";

    DataSources readDataSources() const
    {
        const bool useGps = settings.value(srcGpsKey, true).toBool();
        const int gpsUpdateInterval = settings.value(srcGpsUpdateIntervalKey, 1000).toInt();
        const bool useSound = settings.value(srcSoundKey, false).toBool();
        const bool useVideo = settings.value(srcVideoKey, false).toBool();
        const bool usePics = settings.value(srcPicsKey, true).toBool();
        return DataSources { GpsSourceConfig { useGps, gpsUpdateInterval }, useSound, useVideo, usePics };
    }

    BackupTargets readBackupTargets() const
    {
        const bool useEmail = settings.value(targetEmailEnabledKey, false).toBool();
        const bool useFtp = settings.value(targetFtpEnabledKey, false).toBool();
        const bool useLocalStorage = settings.value(targetLocalEnabledKey, false).toBool();
        const QString email = settings.value(targetEmailKey, "").toString();
        const QString ftpHost = settings.value(targetFtpHostKey, "").toString();
        const QString ftpUser = settings.value(targetFtpUserKey, "").toString();
        const QByteArray ftpPass = settings.value(targetFtpPasswdKey, "").toByteArray();
        const QByteArray localPath = settings.value(targetLocalPathKey, "").toByteArray();

        const EmailTarget emailConfig { useEmail, email };
        const FtpTarget ftpConfig { useFtp, ftpHost, ftpUser, ftpPass };
        const LocalStorage localConfig { useLocalStorage, localPath };

        return BackupTargets { emailConfig, ftpConfig, localConfig };
    }

    SettingsData readCurrent() const
    {
        return SettingsData { readDataSources(), readBackupTargets() };
    }
};

AppSettings::AppSettings(QObject* parent)
    : QObject(parent)
    , priv_(new Priv())
{
}

AppSettings::~AppSettings() = default;

DataSources AppSettings::currentSourceSettings() const
{
    return priv_->readDataSources();
}

BackupTargets AppSettings::currentTargetSettings() const
{
    return priv_->readBackupTargets();
}

SettingsData AppSettings::currentSettings() const
{
    return priv_->readCurrent();
}

void AppSettings::enableStillImageInput(bool enable)
{
    priv_->settings.setValue(Priv::srcPicsKey, enable);
}

void AppSettings::enableSoundInput(bool enable)
{
    priv_->settings.setValue(Priv::srcSoundKey, enable);
}

void AppSettings::enableGpsInput(bool enable)
{
    priv_->settings.setValue(Priv::srcGpsKey, enable);
}

void AppSettings::enableVideoInput(bool enable)
{
    priv_->settings.setValue(Priv::srcVideoKey, enable);
}

void AppSettings::enableLocalStorage(bool enable)
{
    priv_->settings.setValue(Priv::targetLocalEnabledKey, enable);
}

void AppSettings::setLocalStoragePath(const QString& path)
{
    priv_->settings.setValue(Priv::targetLocalPathKey, path.toUtf8()); // todo: encrypt
}

void AppSettings::enableBackupFtp(bool enable)
{
    priv_->settings.setValue(Priv::targetFtpEnabledKey, enable);
}

void AppSettings::setFtpSettings(const FtpTarget& settings)
{
    priv_->settings.setValue(Priv::targetFtpHostKey, settings.host);
    priv_->settings.setValue(Priv::targetFtpUserKey, settings.user);
    priv_->settings.setValue(Priv::targetFtpPasswdKey, settings.passwd);
}
