#include "widget.h"
#include <QTimer>
#include <QPainter>
#include <QMouseEvent>
#include <QKeyEvent>
#include <QTime>
#include <QDebug>

Widget::Widget(QWidget *parent)
    :QWidget(parent)
    ,_use_gpu(true)
    ,maxIter(200)
{
    cuwr::result_t err = cuwr::init();
    if(err==0){
        _gpu_data.reset(new gpu_data);
        _gpu_data->gpu.reset(new cuwr::Gpu(0));
        qDebug() << "GPU initialized: " << QString::fromStdString(_gpu_data->gpu->name());
        _gpu_data->module.loadFile("kernel.ptx");
    } else{
        qDebug() << "GPU init error: " << err;
    }
    startX = -2.5;
    startY = -1.0;
    rangeX = 3.5;
    rangeY = 2.0;
    resize(800,600);
    _image = QImage(2500,2500,QImage::Format_RGB888);
    if (_gpu_data){
        if(_gpu_data->imgData.resize(_image.byteCount()) != 0){
            qDebug() << "CANT ALLOC DATA";
        } else{
            qDebug() << "allocated: " << _image.byteCount()/(1024*1024) << "mb";
        }
    }
    refreshImage();
}

Widget::~Widget(){
    if (_gpu_data){
        _gpu_data.reset();
    }
    cuwr::cleanup();
}

void Widget::paintEvent(QPaintEvent *event){
    QWidget::paintEvent(event);
    QPainter p(this);
    p.drawImage(0,0,_image.scaled(size(),Qt::IgnoreAspectRatio));
    if (_selection.isNull()==false){
        p.drawRect(_selection);
    }
}

static void px_to_coords(const int w, const int h,
                         const int imgW, const int imgH,
                         const float startX, const float startY,
                         const float rangeX, const float rangeY,
                         float * outw, float * outh)
{
    *outw = (rangeX*1.0L*w)/imgW+startX;
    *outh = (rangeY*1.0L*h)/imgH+startY;
}

static void process_pixel(const int i,
                          const int j,
                          unsigned char * pixel_data,
                          const int w,
                          const int h,
                          const int wStep,
                          const float startX,
                          const float startY,
                          const float rangeX,
                          const float rangeY,
                          const int maxIter)
{
    uchar * rgb = pixel_data+wStep*i+j*3;

    float cx,cy;
    px_to_coords(j,i,w,h,startX,startY,
                 rangeX,rangeY,&cx,&cy);
    int iter=0;
    float x0=0;
    float y0=0;
    while (x0*x0+y0*y0<2*2 && iter < maxIter){
        ++iter;
        // x0 = x0^2+C
        const float tmp = x0*x0-y0*y0 + cx;
        y0 = 2*x0*y0+cy;
        x0 = tmp;
    }
    rgb[0] = 255-(255*iter/maxIter);
    rgb[1] = rgb[0];
    rgb[2] = rgb[1];
}

static void gen_fractal(uchar * pixels, const int w, const int h, const int wStep,
                        const float startX, const float startY,
                        const float rangeX, const float rangeY,
                        const int maxIter)
{
    for (int i=0 ; i<h ; ++i){
        for (int j=0 ; j<w ; ++j){
            process_pixel(i,j,pixels,w,h,wStep,startX,startY,rangeX,rangeY,maxIter);
        }
    }
}

void Widget::timeStep(){

}

void Widget::refreshImage(){
    QTime time;
    time.start();
    if (_use_gpu && _gpu_data){
        _image = QImage(_image.width(),_image.height(),_image.format());
        int suggestedBlockSize=0;
        int minGridSize=0;
        cuwr::cuOccupancyMaxPotentialBlockSize(&minGridSize,&suggestedBlockSize,
                                               _gpu_data->module.function("kernel"),
                                               0,0,_image.width()*_image.height());
        const int blockSize = suggestedBlockSize;
        const int gridSize = 512;
        const int totalPixels = _image.width()*_image.height();
        const int pxSingleLaunch = blockSize*gridSize;
        int pxDone = 0;
        int nLaunches=0;
        int w = _image.width();
        int h = _image.height();
        int bpl = _image.bytesPerLine();
        _gpu_data->params.clear();
        _gpu_data->params.setGridSize(blockSize,1);
        _gpu_data->params.setBlockSize(gridSize,1);
        _gpu_data->params.push(&pxDone);
        _gpu_data->params.push(_gpu_data->imgData);
        _gpu_data->params.push(&w);
        _gpu_data->params.push(&h);
        _gpu_data->params.push(&bpl);
        _gpu_data->params.push(&startX);
        _gpu_data->params.push(&startY);
        _gpu_data->params.push(&rangeX);
        _gpu_data->params.push(&rangeY);
        _gpu_data->params.push(&maxIter);
        while (pxDone < totalPixels){
            cuwr::result_t r = cuwr::launch_kernel(_gpu_data->module.function("kernel"),_gpu_data->params);
            ++nLaunches;
            if (r != 0){
                qDebug() << "Error in kernel launch: " << r;
                break;
            } else{
                int toCopy = pxSingleLaunch;
                if (toCopy+pxDone > totalPixels){
                    toCopy = totalPixels-pxDone;
                }
                if (_tmpBuffer.byteCount() != _image.byteCount()){
					_tmpBuffer = QImage(_image.size(), _image.format());
				}
                _gpu_data->imgData.store((void*)_tmpBuffer.bits());
                memcpy((void *)(_image.bits()+pxDone*3),
                                _tmpBuffer.bits()+pxDone*3,
							    toCopy*3);
            }
            pxDone += pxSingleLaunch;
        }
        qDebug() << "gpu processing: " << time.elapsed() << "ms. (launched " << nLaunches << "times)";
    } else{
        gen_fractal(_image.bits(),_image.width(),_image.height(),_image.bytesPerLine(),
                    startX,startY,rangeX,rangeY,maxIter);
        qDebug() << "cpu processing: " << time.elapsed() << "ms.";
    }
    qDebug() << "scale 1:" << 1/(rangeX/3.5);
    update();
}

void Widget::mousePressEvent(QMouseEvent *event){
    QWidget::mousePressEvent(event);
    _selection.setTopLeft(event->pos());
}

void Widget::mouseMoveEvent(QMouseEvent *event){
    QWidget::mouseMoveEvent(event);
    _selection.setBottomRight(event->pos());
    update();
}

void Widget::mouseReleaseEvent(QMouseEvent *event){
    QWidget::mouseReleaseEvent(event);
    float sx, sy, fx, fy;
    px_to_coords(_selection.x(), _selection.y(),
                 width(),height(),
                 startX,startY,rangeX,rangeY,
                 &sx,&sy);
    px_to_coords(event->pos().x(), event->pos().y(),
                 width(),height(),
                 startX,startY,rangeX,rangeY,
                 &fx,&fy);
    _selection = QRect();
    startX = sx;
    startY = sy;
    rangeX = fx-sx;
    rangeY = fy-sy;

    refreshImage();

}

void Widget::keyPressEvent(QKeyEvent *event){
    QWidget::keyPressEvent(event);
    if (event->key() == Qt::Key_Escape){
        this->close();
        return;
    } else if (event->key() == Qt::Key_C){
        _use_gpu = !_use_gpu;
    } else if (event->key() == Qt::Key_Minus){
        maxIter = qBound(10,maxIter-50,maxIter);
    } else if (event->key() == Qt::Key_Plus){
        maxIter = qBound(10,maxIter+50,50000);
    } else if(event->key() == Qt::Key_R){
        maxIter = 200;
        startX = -2.5;
        startY = -1.0;
        rangeX = 3.5;
        rangeY = 2.0;
    }
    refreshImage();
}
