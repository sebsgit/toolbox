#ifndef CUWR_MOTION_ESTIMATOR_H
#define CUWR_MOTION_ESTIMATOR_H

#include "cuwr_img.h"
#include <vector>

namespace cuwr{
    class VectorField{
    public:
        VectorField(const size_t width=0, const size_t height=0){
            this->resize(width,height);
        }
        ~VectorField(){

        }
        void resize(const size_t x, const size_t y, const cuwr_vec2& initValue=cuwr_vec2(0,0)){
            this->data_.resize(x*y,initValue);
            this->size_.x = x;
            this->size_.y = y;
        }
        void resize(const cuwr_dim2& dim){
            this->resize(dim.x,dim.y);
        }
        bool isEmpty() const{
            return this->data_.size() == 0;
        }
        size_t count() const{
            return this->data_.size();
        }
        size_t width() const{ return this->size_.x; }
        size_t height() const{ return this->size_.y; }
        cuwr_dim2 size() const{ return this->size_; }
        std::vector<cuwr_vec2> to_vector() const{
            return data_;
        }
        void set(size_t index,const cuwr_vec2& value){
            this->data_[index] = value;
        }
        cuwr_vec2 get(size_t row, size_t col) const{
            return this->data_[ col + row*size_.x ];
        }

    private:
        std::vector<cuwr_vec2> data_;
        cuwr_dim2 size_;

        friend class MotionEstimator;
    };

    /*
        motion estimator
        compares adjacent rectangular blocks of pixels on two images to
        estimate the motion of image elements
     */
    class MotionEstimator{
    public:
        MotionEstimator(size_t blockSize=16, int searchWindow=8)
            :blockSize_(blockSize > 0 ? blockSize : 16)
            ,searchWindow_(searchWindow > 0 ? searchWindow : 8)
        {
            cuwr::result_t err = module.load("cuwr_motion.ptx");
            if (err == cuwr::CUDA_SUCCESS_){
                this->calc_mad = module.function("cuwr_MAD",&err);
            }
            if (err != cuwr::CUDA_SUCCESS_)
                throw cuwr::Exception(err);
        }
        VectorField estimateMotionField(const cuwr::Image& image1,
                                        const cuwr::Image& image2)
        {

            struct mad_result_t{
                float madValue=std::numeric_limits<float>::max();
                cuwr_dim2 offset;
            };

            VectorField result;
            if (image1.size() == image2.size()){
                result.resize( ((image1.width()+blockSize_)-1)/blockSize_,
                               ((image1.height()+blockSize_)-1)/blockSize_ );
                if (result.isEmpty()==false){
                    std::vector<mad_result_t> perBlockResult;
                    perBlockResult.resize(result.count());
                    cuwr::DeviceValue<cuwr_dim2> off_dev = cuwr_dim2(0,0);
                    cuwr::DeviceArray<float, cuwr::DeviceMemPinnedAllocator> mads;
                    mads.resize(result.count());
                    cuwr::KernelLaunchParams params;
                    params.autodetect(image1.size(),blockSize_);
                    params.setSharedMemoryCount(sizeof(float));
                    image1.pushData(params);
                    image1.pushHeader(params);
                    image2.pushData(params);
                    image2.pushHeader(params);
                    params.push(off_dev);
                    params.push(mads);

                    for (int i=-searchWindow_+1 ; i<searchWindow_ ; ++i){
                        for (int j=-searchWindow_+1 ; j<searchWindow_ ; ++j){
                            off_dev = cuwr_dim2(i,j);
                            if (cuwr::result_t r = cuwr::launch_kernel(this->calc_mad,params)){
                                throw Exception(r);
                            }
                            cuwr::cuStreamSynchronize(0);
                            //TODO: move this to kernel
                            float * mad_values = (float*)mads.hostAddress();
                            for (size_t k=0 ; k<mads.count() ; ++k){
                                if (mad_values[k] < perBlockResult[k].madValue){
                                    perBlockResult[k].madValue = mad_values[k];
                                    perBlockResult[k].offset = cuwr_dim2(i,j);
                                }
                            }
                        }
                    }
                    for (size_t i=0 ; i<perBlockResult.size() ; ++i){
                        const mad_result_t b = perBlockResult[i];
                        if (b.madValue > 0.0){
                            result.set(i,cuwr_vec2(b.offset.x,b.offset.y));
                      //     std::cout << "block moved: " << b.blockIndex << " " << b.madValue << ", "
                       //           << b.offset.x << "x" << b.offset.y << "\n";
                        }
                    }
                }
            }
            return result;
        }

    private:
        size_t blockSize_;
        int searchWindow_;
        cuwr::Module module;
        cuwr::function_t calc_mad;
    };
}

#endif // CUWR_MOTION_ESTIMATOR_H

