#include "AppSettings.h"
#include <QDebug>
#include <QSettings>

#include "clc/clc_encrypt.h"

class AppSettings::Priv {
public:
    QSettings settings { "BackupAppDomain", "BackupApp" };
    QString pinCode;

    constexpr static const char* const pinCodeKey = "src/PIN";

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

    static QByteArray padTo16(const QByteArray& input, int* numAdded)
    {
        if (input.size() % 16 == 0) {
            *numAdded = 0;
            return input;
        }
        const int missing = 16 - input.size() % 16;
        *numAdded = missing;
        return input + QByteArray(missing, '\0');
    }

    QByteArray encrypt(const QByteArray& input) const
    {
        int extraChars = 0;
        const QByteArray s = padTo16(input, &extraChars);
        QByteArray result = s;
        const QByteArray pwd = pinCode.toLatin1();
        clc_aes_key key;
        clc_aes_init_key(&key,
            reinterpret_cast<const unsigned char*>(pwd.constData()),
            static_cast<size_t>(pwd.size()),
            clc_cipher_type::CLC_AES_256);
        clc_aes_encrypt(reinterpret_cast<unsigned char*>(result.data()),
            reinterpret_cast<const unsigned char*>(s.constData()),
            static_cast<size_t>(s.size()),
            key,
            clc_cipher_type::CLC_AES_256);
        return result;
    }

    QByteArray decrypt(const QByteArray& s) const
    {
        QByteArray result = s;
        const QByteArray pwd = pinCode.toLatin1();
        clc_aes_key key;
        clc_aes_init_key(&key,
            reinterpret_cast<const unsigned char*>(pwd.constData()),
            static_cast<size_t>(pwd.size()),
            clc_cipher_type::CLC_AES_256);
        clc_aes_decrypt(reinterpret_cast<unsigned char*>(result.data()),
            reinterpret_cast<const unsigned char*>(s.constData()),
            static_cast<size_t>(s.size()),
            key,
            clc_cipher_type::CLC_AES_256);
        return result;
    }

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
        const QByteArray ftpHost = settings.value(targetFtpHostKey, "").toByteArray();
        const QByteArray ftpUser = settings.value(targetFtpUserKey, "").toByteArray();
        const QByteArray ftpPass = settings.value(targetFtpPasswdKey, "").toByteArray();
        const QByteArray localPath = settings.value(targetLocalPathKey, "").toByteArray();

        qDebug() << "Data in settings: " << ftpHost << ftpUser << ftpPass;

        const EmailTarget emailConfig { useEmail, email };
        const FtpTarget ftpConfig { useFtp, decrypt(ftpHost), decrypt(ftpUser), decrypt(ftpPass) };
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

bool AppSettings::hasValidPINCode() const noexcept
{
    return !priv_->pinCode.isEmpty();
}

bool AppSettings::isPINCodeValid(const QString& key) const
{
    const QByteArray savedKey = priv_->settings.value(Priv::pinCodeKey).toByteArray();
    if (savedKey.isEmpty()) {
        return false;
    }
    return qCompress(key.toLatin1()) == savedKey;
}

bool AppSettings::isPINSaved() const
{
    return priv_->settings.value(Priv::pinCodeKey).isValid();
}

void AppSettings::setPINCode(const QString& pinCode)
{
    const QByteArray pinCodeMangled = qCompress(pinCode.toLatin1());
    priv_->pinCode = pinCode;
    priv_->settings.setValue(Priv::pinCodeKey, pinCodeMangled);
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
    priv_->settings.setValue(Priv::targetFtpHostKey, priv_->encrypt(settings.host.toLatin1()));
    priv_->settings.setValue(Priv::targetFtpUserKey, priv_->encrypt(settings.user.toLatin1()));
    priv_->settings.setValue(Priv::targetFtpPasswdKey, priv_->encrypt(settings.passwd));
}
