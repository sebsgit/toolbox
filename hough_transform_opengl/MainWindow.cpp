#include "MainWindow.h"
#include "ui_MainWindow.h"

#include <QTimer>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    QTimer::singleShot(0, [this]() {
       resize(480, 640);
    });
}

MainWindow::~MainWindow()
{
    delete ui;
}

