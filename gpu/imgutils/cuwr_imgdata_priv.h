#ifndef CUWR_IMGDATA_PRIV_H
#define CUWR_IMGDATA_PRIV_H

#ifdef __cplusplus
extern "C" {
#endif

struct cuwr_image_kernel_data_t_{
    size_t width;           /* width in pixels */
    size_t height;          /* height in pixels */
    size_t widthStep;       /* scan line size */
    size_t bpp;             /* bytes per pixel */
};

typedef struct cuwr_image_kernel_data_t_ cuwr_image_kernel_data_t;

#ifdef __cplusplus
}
#endif

#endif
