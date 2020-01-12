#ifndef PINSELECTIONWIDGET_H
#define PINSELECTIONWIDGET_H

#include <QWidget>
#include <memory>

#include "AppSettings.h"

namespace Ui {
class PINSelectionWidget;
}

class PINSelectionWidget : public QWidget {
    Q_OBJECT

public:
    explicit PINSelectionWidget(AppSettings& settings, QWidget* parent = nullptr);
    ~PINSelectionWidget();

signals:
    void validPINEntered();
    void statusMessage(const QString& message);

private:
    std::unique_ptr<Ui::PINSelectionWidget> ui_;
};

#endif // PINSELECTIONWIDGET_H
