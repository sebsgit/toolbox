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
    QImage image("test.jpg");
    cuwr::Image cimg = cuwr::Image::fromQImage(image.convertToFormat(QImage::Format_RGB888));
    cimg.swapRgb();
    cimg.sync();
    QImage image2 = cimg.toQImage();
    image2.save("proc.jpg");
	cuwr::cleanup();
	return 0;
}
