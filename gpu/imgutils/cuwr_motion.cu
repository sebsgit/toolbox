#include "cuwr_imgdata_priv.h"

typedef unsigned char uchar;

class CoordCalc{
public:
    __device__ CoordCalc(const dim3& thIdx,
                         const dim3& blockId,
                         const dim3& blockDim,
                         const dim3& gridDim)
        :threadIdx_(thIdx)
        ,blockIdx_(blockId)
        ,blockDim_(blockDim)
        ,gridDim_(gridDim)
    {
        const size_t idInBlock = threadIdx_.x + threadIdx_.y*blockDim_.x;
        const size_t blockIdInGrid = blockIdx_.x + blockIdx_.y*gridDim_.x;
        this->pixelId_ = idInBlock + blockIdInGrid*blockDim_.x*blockDim_.y;
    }
    __device__ size_t pixelId() const{
        return this->pixelId_;
    }
    __device__ void getCoords(const size_t width, size_t * row, size_t * col) const{
        *col = (pixelId_) % width;
        *row = (pixelId_) / width;
    }
    __device__ size_t dataOffset(const cuwr_image_kernel_data_t * header) const{
        const size_t col_id = (pixelId_) % header->width;
        const size_t row_id = (pixelId_) / header->width;
        return col_id*header->bpp + row_id*header->widthStep;
    }
    __device__ size_t dataOffset(const size_t row, const size_t col, const cuwr_image_kernel_data_t * header) const{
        return col*header->bpp + row*header->widthStep;
    }
private:
    const dim3& threadIdx_;
    const dim3& blockIdx_;
    const dim3& blockDim_;
    const dim3& gridDim_;
    size_t pixelId_;
};

extern "C"{

/* calculate MAD of two images (mean absolute difference) per block
   output: array of MAD values (one number per block)

   this function should be launched with assigned shared memory for the block MAD value
   thread block size should match the logical image block size
*/
__global__ void cuwr_MAD(const uchar * image1, const cuwr_image_kernel_data_t * header1,
                         const uchar * image2, const cuwr_image_kernel_data_t * header2,
                         const cuwr_dim2 * offsets, float * output)
            {
                __shared__ float block_mad_value;
                const float badValue = 999999.0f;
                float thisThreadMad = badValue;
                const CoordCalc calc(threadIdx,blockIdx,blockDim,gridDim);
                int row = threadIdx.y + blockIdx.y*blockDim.y;
                int col= threadIdx.x + blockIdx.x*blockDim.x;
                if (row < header1->height && col < header1->width){
                    const uchar * pxIn1 = image1 + calc.dataOffset(row,col,header1);
                    row -= offsets->y;
                    col -= offsets->x;
                    if (row >= 0 && col >=0 && row < header2->height && col < header2->width){
                        const uchar * pxIn2 = image2 + calc.dataOffset(row,col,header2);
                        thisThreadMad = abs((float)pxIn1[0]-(float)pxIn2[0]);
                        if (header1->bpp > 1){
                            thisThreadMad += abs((float)pxIn1[1]-(float)pxIn2[1]);
                            thisThreadMad += abs((float)pxIn1[2]-(float)pxIn2[2]);
                        }
                    } else{
                        thisThreadMad = 255.0f;
                        if (header1->bpp > 1){
                            thisThreadMad += 255.0f;
                            thisThreadMad += 255.0f;
                        }
                    }
                }
                if (thisThreadMad > 0.0f && thisThreadMad < badValue)
                    atomicAdd(&block_mad_value,thisThreadMad);
                __syncthreads();
                if (threadIdx.x==0 && threadIdx.y==0){
                    block_mad_value /= blockDim.x*blockDim.y;
                    output[ blockIdx.x + blockIdx.y*gridDim.x ] = block_mad_value;
                }
            }

/* calculate absolute value of difference between [image1] and [image2+offset]
    that is outp(x,y) = abs(image1(x,y) - image2(x+off.x, y+off.y))
    equal image sizes are assumed
    coordinates out of bounds are assumed to have value = 0
*/
__global__ void cuwr_diff_image( const uchar * image1, const cuwr_image_kernel_data_t * header1,
                                 const uchar * image2, const cuwr_image_kernel_data_t * header2,
                                 const cuwr_dim2 * offsets,
                                 uchar * output, const cuwr_image_kernel_data_t * header_out)
            {
                const CoordCalc calc(threadIdx,blockIdx,blockDim,gridDim);
                const size_t id = calc.pixelId();
                if (id < header1->width*header1->height){
                    size_t row, col;
                    calc.getCoords(header1->width,&row,&col);
                    const uchar * pxIn1 = image1 + calc.dataOffset(header1);
                    uchar * pxOut = output + calc.dataOffset(header_out);
                    row -= offsets->y;
                    col -= offsets->x;
                    if (row < header2->height && col < header2->width){
                        const uchar * pxIn2 = image2 + calc.dataOffset(row,col,header2);
                        pxOut[0] = abs(pxIn1[0]-pxIn2[0]);
                        if (header1->bpp > 1){
                            pxOut[1] = abs(pxIn1[1]-pxIn2[1]);
                            pxOut[2] = abs(pxIn1[2]-pxIn2[2]);
                        }
                    } else{
                        pxOut[0] = pxIn1[0];
                        if (header1->bpp > 1){
                            pxOut[1] = pxIn1[1];
                            pxOut[2] = pxIn1[2];
                        }
                    }
                }
            }

}
