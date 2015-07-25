#include <iostream>
#include <string>
#include "cuwr_img.h"
#include "cuwr_motion_estimator.h"
#include <QDebug>
#include <QTime>

int main(int argc, char ** argv){
	if (cuwr::init() != 0){
		std::cout << "cant init cuda!\n";
		exit(0);
	}
    cuwr::Gpu * gpu = new cuwr::Gpu();
    gpu->makeCurrent();
    size_t w,h;
    cuwr::Image::maxImageSize(*gpu,&w,&h);
    std::cout << "max image size: " << w << " x " << h << '\n';
    QImage image("test.jpg");
    cuwr::Image cimg = cuwr::Image::fromQImage(image.convertToFormat(QImage::Format_RGB888));
    cimg.setAutoSync();
    cimg.swapRgb();
    QImage image2 = cimg.toQImage();
    image2.save("proc.jpg");
    cuwr::Image part = cimg.copy(cimg.width()/2,cimg.height()/2,cimg.width()/2,cimg.height()/2);
    QImage image3 = part.toQImage();
    image3.save("part.jpg");

    {
        QImage frame1("frame1.png");
        QImage frame2("frame2.png");

        if (argc > 2){
            frame1.load(argv[1]);
            frame2.load(argv[2]);
            frame1 = frame1.convertToFormat(QImage::Format_RGB888);
            frame2 = frame2.convertToFormat(QImage::Format_RGB888);
        }

        const int searchWindow = 7;
        const int blockSize = 16;
        cuwr::MotionEstimator motion(blockSize,searchWindow);
        cimg = cuwr::Image(64,64,cuwr::Format_Rgb24);
        cimg.setAutoSync();
        cuwr::Image cimg2;
        cimg = cuwr::Image::fromQImage(frame1);
        cimg2 = cuwr::Image::fromQImage(frame2);

        QTime t; t.start();
        cuwr::VectorField vec = motion.estimateMotionField(cimg,cimg2);
        qDebug() << t.elapsed();
        const float maxVecLength = sqrt(2.0f*searchWindow*searchWindow);
        QImage vecFieldImage(vec.width(),vec.height(),QImage::Format_RGB888);
        for (size_t i = 0 ; i < vec.width() ; ++i ){
            for (size_t j = 0 ; j < vec.height() ; ++j ){
                const cuwr_vec2 v = vec.get(j,i);
                const int color = (255*v.length())/maxVecLength;
                vecFieldImage.setPixel(i,j,qRgb(color,0,0));
            }
        }
        vecFieldImage.scaledToHeight(cimg.height()).save("motion.jpg");

    }

    delete gpu;
	cuwr::cleanup();
	return 0;
}
