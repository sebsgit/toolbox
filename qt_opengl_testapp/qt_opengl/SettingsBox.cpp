#include "SettingsBox.h"
#include "ui_SettingsBox.h"

#include <QDoubleValidator>

SettingsBox::SettingsBox(QWidget *parent) :
    QGroupBox(parent),
    ui(new Ui::SettingsBox)
{
    ui->setupUi(this);
    auto createValidator = [this]() {
        return new QDoubleValidator(-100.0, 100.0, 8, this);
    };
    for (auto & line_edit : ui->group_distortion->findChildren<QLineEdit*>())
    {
        line_edit->setValidator(createValidator());
        QObject::connect(line_edit, &QLineEdit::editingFinished,
        [this]() {
            emit distortionSettingsChanged(currentDistortionParameters());
        });
    }

    QObject::connect(ui->rotation_slider, &QSlider::valueChanged,
    [this](int angle)
    {
        emit rotationAngleChanged(qreal(angle));
    });

    QObject::connect(ui->spinBox_grid_x, QOverload<int>::of(&QSpinBox::valueChanged),
    [this](int)
    {
        emit gridSizeChanged(currentGridSize());
    });

    QObject::connect(ui->spinBox_grid_y, QOverload<int>::of(&QSpinBox::valueChanged),
    [this](int)
    {
        emit gridSizeChanged(currentGridSize());
    });

    QObject::connect(ui->checkBox_enable_grid, &QCheckBox::stateChanged,
    [this](auto)
    {
       emit overlayGridChanged(ui->checkBox_enable_grid->isChecked());
    });
}

SettingsBox::~SettingsBox()
{
    delete ui;
}

DistortionParameters SettingsBox::currentDistortionParameters() const
{
    DistortionParameters result;
    result.k1 = ui->edit_k1->text().toDouble();
    result.k2 = ui->edit_k2->text().toDouble();
    result.k3 = ui->edit_k3->text().toDouble();
    result.fx = ui->edit_fx->text().toDouble();
    result.fy = ui->edit_fy->text().toDouble();
    result.cx = ui->edit_cx->text().toDouble();
    result.cy = ui->edit_cy->text().toDouble();
    return result;
}

QSize SettingsBox::currentGridSize() const
{
    return QSize(ui->spinBox_grid_x->value(), ui->spinBox_grid_y->value());
}
