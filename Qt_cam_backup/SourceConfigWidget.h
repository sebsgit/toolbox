#ifndef SOURCECONFIGWIDGET_H
#define SOURCECONFIGWIDGET_H

#include <QWidget>
#include <memory>

#include "AppSettings.h"

class SourceConfigWidget : public QWidget {
    Q_OBJECT

public:
    explicit SourceConfigWidget(AppSettings& settings, QWidget* parent = nullptr);
    ~SourceConfigWidget();

private:
    class Priv;
    std::unique_ptr<Priv> priv_;
};

#endif // SOURCECONFIGWIDGET_H
