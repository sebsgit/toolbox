#include "MainWindow.h"
#include "ui_MainWindow.h"

#include <QTimer>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    const int preferred_width {400};

    QObject::connect(ui->openGLWidget, &DisplayWindow::cpuImageRendered,
        [this](QImage img)
        {
           ui->cpu_image->setPixmap(QPixmap::fromImage(img));
        }
    );
    QObject::connect(ui->openGLWidget, &DisplayWindow::opencvImageRendered,
        [this](QImage img)
        {
           ui->opencv_image->setPixmap(QPixmap::fromImage(img));
        }
    );

    QObject::connect(ui->settings, &SettingsBox::rotationAngleChanged,
                     ui->openGLWidget, &DisplayWindow::setRotation);

    QObject::connect(ui->settings, &SettingsBox::distortionSettingsChanged,
                     ui->openGLWidget, &DisplayWindow::setDistortion);

    QObject::connect(ui->settings, &SettingsBox::gridSizeChanged,
    [this](QSize s)
    {
       ui->openGLWidget->setGridSize(s.width(), s.height());
    });

    QObject::connect(ui->settings, &SettingsBox::overlayGridChanged,
                     ui->openGLWidget, &DisplayWindow::setDisplayGrid);

    QTimer::singleShot(0, [this]() {
       auto img {QImage(":/images/lenna.png")};
       ui->openGLWidget->loadImage(img);
       ui->openGLWidget->setGridSize(ui->settings->currentGridSize().width(), ui->settings->currentGridSize().height());
       resize(img.width() * 3, img.height() * 1.2);
    });
}

MainWindow::~MainWindow()
{
    delete ui;
}

