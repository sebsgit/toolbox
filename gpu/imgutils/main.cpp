#include <iostream>
#include <string>
#include "cuwr_img.h"
#include <QDebug>

int main(){
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

    delete gpu;
	cuwr::cleanup();
	return 0;
}
