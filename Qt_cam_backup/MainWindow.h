#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <memory>

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
    MainWindow(QWidget* parent = nullptr);
    ~MainWindow();

private:
    void setAsCentral(QWidget* widget);

private slots:
    void hanleDebugMessage(const QString& message);

private:
    class Priv;
    std::unique_ptr<Priv> priv_;
};
#endif // MAINWINDOW_H
