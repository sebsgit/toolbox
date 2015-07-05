extern "C"{

__global__ void kernel(const int startPixel,
                      unsigned char * pixel_data,
					  const int w,
					  const int h,
					  const int wStep,
                      const double startX,
                      const double startY,
                      const double rangeX,
                      const double rangeY,
					  const int maxIter)
{
    const int id = threadIdx.x + blockDim.x*blockIdx.x + startPixel;
	if (id<h*w){
		const int j = id%w;
		const int i = id/w;
        unsigned char * rgb = pixel_data+wStep*i+j*3;
		
        const double cx = (rangeX*1.0*j)/w+startX;
        const double cy = (rangeY*1.0*i)/h+startY;
		
		int iter=0;
        double x0=0;
        double y0=0;
		while (x0*x0+y0*y0<2*2 && iter < maxIter){
			++iter;
			// x0 = x0^2+C
            const double tmp = x0*x0-y0*y0 + cx;
			y0 = 2*x0*y0+cy;
			x0 = tmp;
		}

        const int color = 255 - 255*(iter*1.0/maxIter);
        rgb[0] = color;
        rgb[1] = color;//;rgb[0];
        rgb[2] = color;
	}
}

}
