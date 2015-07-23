#ifndef CUWR_IMGDATA_PRIV_H
#define CUWR_IMGDATA_PRIV_H

struct cuwr_image_kernel_data_t{
    size_t width;           /* width in pixels */
    size_t height;          /* height in pixels */
    size_t widthStep;       /* scan line size */
    size_t bpp;             /* bytes per pixel */
    cuwr_image_kernel_data_t(size_t w=0, size_t h=0,
                             size_t ws=0, size_t bp=0)
        :width(w)
        ,height(h)
        ,widthStep(ws)
        ,bpp(bp)
    {
    }
};

#endif
