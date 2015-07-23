#include "cuwr_imgdata_priv.h"

/*
  image processing gpu utilities
  this module is intended to internal use in cuwr::Image class
*/

extern "C"
{

/* linear mapping of thread position in grid to a pixel number */
__device__ size_t px_id(const dim3 threadIdx,
                        const dim3 blockIdx,
                        const dim3 blockDim,
                        const dim3 gridDim)
            {
                const size_t idInBlock = threadIdx.x + threadIdx.y*blockDim.x;
                const size_t blockIdInGrid = blockIdx.x + blockIdx.y*gridDim.x;
                return idInBlock + blockIdInGrid*blockDim.x*blockDim.y;
            }
/* find data pointer offset for pixel number pxId */
__device__ int px_offset(const size_t pxId,
                         const cuwr_image_kernel_data_t * header)
            {
                const int col_id = pxId % header->width;
                const int row_id = pxId / header->width;
                return col_id*header->bpp + row_id*header->widthStep;
            }

/* sets pixel value to specified [r,g,b] tripple.
    in case of grayscale images, only 'r' value is used
 */
__global__ void cuwr_set_pixels(unsigned char * data,
                                const cuwr_image_kernel_data_t * header,
                                const size_t offset,
                                const unsigned char r,
                                const unsigned char g,
                                const unsigned char b)
            {
                const size_t pxId = px_id(threadIdx,blockIdx,blockDim,gridDim) + offset;
                if (pxId < header->width*header->height){
                    unsigned char * pxDataPtr = data + px_offset(pxId,header);
                    pxDataPtr[0] = r;
                    if (header->bpp > 1){
                        pxDataPtr[1] = g;
                        pxDataPtr[2] = b;
                    }
                }
            }

/* swaps r<->b channels */
__global__ void cuwr_swap_rgb(unsigned char * data,
                              const cuwr_image_kernel_data_t * header,
                              const size_t offset
                            )
			{
                const size_t pxId = px_id(threadIdx,blockIdx,blockDim,gridDim) + offset;
                if (pxId < header->width*header->height){
                    unsigned char * pxDataPtr = data + px_offset(pxId,header);
                    const unsigned char r = pxDataPtr[0];
                    pxDataPtr[0] = pxDataPtr[2];
                    pxDataPtr[2] = r;
                }
            }
	
}
