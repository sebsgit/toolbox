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
        void resize(const size_t x, const size_t y){
            this->data_.resize(x*y);
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
            return this->data_.count();
        }
        size_t width() const{ return this->size_.x; }
        size_t height() const{ return this->size_.y; }
        cuwr_dim2 size() const{ return this->size_; }
        std::vector<cuwr_vec2> to_vector() const{
            std::vector<cuwr_vec2> result;
            result.resize(this->count());
            this->data_.store(&result[0]);
            return result;
        }
    private:
        cuwr::DeviceArray<cuwr_vec2> data_;
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
        MotionEstimator(size_t blockSize=16)
            :blockSize_(blockSize > 0 ? blockSize : 16)
        {
            cuwr::result_t err = module.load("cuwr_motion.ptx");
            if (err == cuwr::CUDA_SUCCESS_){
                this->motion_estimate = module.function("cuwr_motion_estimate",&err);
            }
            if (err != cuwr::CUDA_SUCCESS_)
                throw cuwr::Exception(err);
        }
        VectorField estimateMotionField(const cuwr::Image& image1,
                                        const cuwr::Image& image2)
        {
            VectorField result;
            if (image1.size() == image2.size()){
                result.resize( ((image1.width()+blockSize_)-1)/blockSize_,
                               ((image1.height()+blockSize_)-1)/blockSize_ );
                if (result.isEmpty()==false){
                    cuwr::KernelLaunchParams params;
                    params.autodetect(result.count());
                    params.push(result.data_);
                    params.push(&result.size_);
                    image1.pushData(params);
                    image1.pushHeader(params);
                    image2.pushData(params);
                    image2.pushHeader(params);
                    if (cuwr::result_t r = cuwr::launch_kernel(this->motion_estimate,params)){
                        throw Exception(r);
                    }
                }
            }
            return result;
        }
    private:
        size_t blockSize_;
        cuwr::Module module;
        cuwr::function_t motion_estimate;
    };
}

#endif // CUWR_MOTION_ESTIMATOR_H

