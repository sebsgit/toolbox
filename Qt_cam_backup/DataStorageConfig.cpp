#include "DataStorageConfig.h"
#include "ui_DataStorageConfig.h"

#include <QFileDialog>

class DataStorageConfig::Priv {
public:
    Priv(AppSettings& sett)
        : settings(sett)
    {
    }
    AppSettings& settings;
    Ui::DataStorageConfig ui = {};
};

DataStorageConfig::DataStorageConfig(AppSettings& settings, QWidget* parent)
    : QWidget(parent)
    , priv_(new Priv(settings))
{
    priv_->ui.setupUi(this);

    const SettingsData s = settings.currentSettings();
    priv_->ui.groupBox_email->setChecked(s.backup.email.enabled);
    priv_->ui.lineEdit_email->setText(s.backup.email.path);

    priv_->ui.groupBox_local->setChecked(s.backup.local.enabled);

    priv_->ui.groupBox_ftp->setChecked(s.backup.ftp.enabled);
    priv_->ui.lineEdit_ftp_host->setText(s.backup.ftp.host);
    priv_->ui.lineEdit_ftp_user->setText(s.backup.ftp.user);
    priv_->ui.lineEdit_ftp_passwd->setText(s.backup.ftp.passwd);

    QObject::connect(priv_->ui.groupBox_local, &QGroupBox::toggled, this, &DataStorageConfig::onGroupBoxLocalChecked);
    QObject::connect(priv_->ui.pushButon_browse_local_storage, &QToolButton::clicked, this, &DataStorageConfig::selectLocalStoragePath);

    QObject::connect(priv_->ui.groupBox_ftp, &QGroupBox::toggled, &settings, &AppSettings::enableBackupFtp);
    auto storeFtpSettings = [&]() {
        settings.setFtpSettings(ftpSettings());
    };
    QObject::connect(priv_->ui.lineEdit_ftp_host, &QLineEdit::editingFinished, storeFtpSettings);
    QObject::connect(priv_->ui.lineEdit_ftp_user, &QLineEdit::editingFinished, storeFtpSettings);
    QObject::connect(priv_->ui.lineEdit_ftp_passwd, &QLineEdit::editingFinished, storeFtpSettings);

    //TODO: add option to test FTP upload directly in settings
}

DataStorageConfig::~DataStorageConfig() = default;

FtpTarget DataStorageConfig::ftpSettings() const noexcept
{
    return FtpTarget { priv_->ui.groupBox_ftp->isChecked(),
        priv_->ui.lineEdit_ftp_host->text(),
        priv_->ui.lineEdit_ftp_user->text(),
        priv_->ui.lineEdit_ftp_passwd->text().toLatin1() };
}

void DataStorageConfig::onGroupBoxLocalChecked(bool checked)
{
    priv_->settings.enableLocalStorage(checked);
}

void DataStorageConfig::selectLocalStoragePath()
{
    QString path = QFileDialog::getExistingDirectory(this, tr("select storage path"));
    if (!path.isEmpty())
        priv_->settings.setLocalStoragePath(path);
}
