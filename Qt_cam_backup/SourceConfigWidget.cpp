#include "SourceConfigWidget.h"
#include "ui_SourceConfigWidget.h"

class SourceConfigWidget::Priv {
public:
    explicit Priv(AppSettings& s) noexcept
        : settings(s)
    {
    }

    Ui::SourceConfigWidget ui;
    AppSettings& settings;
};

SourceConfigWidget::SourceConfigWidget(AppSettings& settings, QWidget* parent)
    : QWidget(parent)
    , priv_(new Priv(settings))
{
    priv_->ui.setupUi(this);

    const DataSources s = settings.currentSourceSettings();

    priv_->ui.checkBox_gps->setChecked(s.gps.enabled);
    priv_->ui.checkBox_video->setChecked(s.video);
    priv_->ui.checkBox_sound->setChecked(s.sound);
    priv_->ui.checkBox_pictures->setChecked(s.pictures);

    auto bindCheckbox = [&settings](auto box, auto methodPtr) {
        QObject::connect(box, &QCheckBox::toggled, &settings, methodPtr);
    };
    bindCheckbox(priv_->ui.checkBox_gps, &AppSettings::enableGpsInput);
    bindCheckbox(priv_->ui.checkBox_video, &AppSettings::enableVideoInput);
    bindCheckbox(priv_->ui.checkBox_sound, &AppSettings::enableSoundInput);
    bindCheckbox(priv_->ui.checkBox_pictures, &AppSettings::enableStillImageInput);
}

SourceConfigWidget::~SourceConfigWidget() = default;
