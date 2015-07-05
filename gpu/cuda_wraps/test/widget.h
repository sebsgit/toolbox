#ifndef WIDGET_H
#define WIDGET_H

#include <QWidget>
#include <QImage>
#include <iostream>
#include <QScopedPointer>
#include "cuwrap.h"

class Widget : public QWidget{
    Q_OBJECT
public:
    Widget(QWidget * parent=nullptr);
    ~Widget();
protected:
    void paintEvent(QPaintEvent * event);
    void mousePressEvent(QMouseEvent * event);
    void mouseMoveEvent(QMouseEvent * event);
    void mouseReleaseEvent(QMouseEvent * event);
    void keyPressEvent(QKeyEvent * event);
private slots:
    void timeStep();
    void refreshImage();
private:
    QImage _image;
    QRect _selection;
    double startX, startY, rangeX, rangeY;

    struct gpu_data{
        QScopedPointer<cuwr::Gpu> gpu;
        cuwr::DevicePtr<unsigned char> imgData;
        cuwr::KernelLaunchParams params;
        cuwr::Module module;
    };
    QScopedPointer<struct gpu_data> _gpu_data;
    bool _use_gpu;
    int maxIter;
};

#endif // WIDGET_H

