#ifndef SETTINGSBOX_H
#define SETTINGSBOX_H

#include <QGroupBox>
#include "DistortionParameters.h"

namespace Ui {
    class SettingsBox;
}

class SettingsBox : public QGroupBox
{
    Q_OBJECT

public:
    explicit SettingsBox(QWidget *parent = nullptr);
    ~SettingsBox();

    DistortionParameters currentDistortionParameters() const;
    QSize currentGridSize() const;

signals:
    void distortionSettingsChanged(DistortionParameters);
    void rotationAngleChanged(qreal);
    void gridSizeChanged(QSize);
    void overlayGridChanged(bool);

private:
    Ui::SettingsBox *ui;
};

#endif // SETTINGSBOX_H
