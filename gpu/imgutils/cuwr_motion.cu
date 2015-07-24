#include "cuwr_imgdata_priv.h"

typedef unsigned char uchar;

extern "C"{

/* find element coordinates in rectangular grid according to thread position and grid width */
__device__ void block_coords(const dim3 threadIdx,
                            const dim3 blockIdx,
                            const dim3 blockDim,
                            const dim3 gridDim,
                            const size_t width,
                            size_t * row_id,
                            size_t * col_id)
            {
                const size_t idInBlock = threadIdx.x + threadIdx.y*blockDim.x;
                const size_t blockIdInGrid = blockIdx.x + blockIdx.y*gridDim.x;
                const size_t pxId = idInBlock + blockIdInGrid*blockDim.x*blockDim.y;
                *col_id = pxId % width;
                *row_id = pxId / width;
            }

/*
 basic motion estimator
 compares adjacent rectangular blocks to estimate the vector field
 //todo different algorithms
 */
__global__ void cuwr_motion_estimate( cuwr_vec2 * output,
                                      const cuwr_dim2 fieldSize,
                                      const uchar * image1,
                                      const cuwr_image_kernel_data_t * header_image1,
                                      const uchar * image2,
                                      const cuwr_image_kernel_data_t * header_image2)
            {
                size_t row,col;
                block_coords(threadIdx,blockIdx,blockDim,gridDim,fieldSize.x,&row,&col);
                if (row < fieldSize.y && col < fieldSize.x){
                    cuwr_vec2 * thisVec = output + col + row*fieldSize.x;
                    thisVec->x = col;
                    thisVec->y = row;
                }
            }

}
