#include "PINSelectionWidget.h"
#include "ui_PINSelectionWidget.h"

#include <QTimer>

PINSelectionWidget::PINSelectionWidget(AppSettings& settings, QWidget* parent)
    : QWidget(parent)
    , ui_(new Ui::PINSelectionWidget)
{
    ui_->setupUi(this);
    if (settings.isPINSaved()) {
        ui_->lineEdit_PIN_2->setText("1111");
    }
    auto checkPIN = [this, &settings]() {
        if (settings.isPINSaved()) {
            if (settings.isPINCodeValid(ui_->lineEdit_PIN_1->text())) {
                settings.setPINCode(ui_->lineEdit_PIN_1->text());
                emit statusMessage("Settings unlocked");
                emit validPINEntered();
            } else {
                emit statusMessage("Entered PIN code is not valid");
            }
        } else {
            const auto pinCode = ui_->lineEdit_PIN_1->text();
            if (!pinCode.isEmpty() && pinCode == ui_->lineEdit_PIN_2->text()) {
                settings.setPINCode(pinCode);
                emit statusMessage("Settings unlocked");
                emit validPINEntered();
            } else {
                emit statusMessage("PIN code must match on both lines and not be empty");
            }
        }
    };
    QObject::connect(ui_->lineEdit_PIN_1, &QLineEdit::editingFinished, checkPIN);
    QObject::connect(ui_->lineEdit_PIN_2, &QLineEdit::editingFinished, checkPIN);

#ifndef Q_OS_WIN
    QTimer::singleShot(0, [this]() {
        QEvent event(QEvent::RequestSoftwareInputPanel);
        ui_->lineEdit_PIN_1->setFocus();
        qApp->sendEvent(ui_->lineEdit_PIN_1, &event);
    });
#endif
}

PINSelectionWidget::~PINSelectionWidget() = default;
