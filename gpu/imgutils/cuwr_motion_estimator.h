#ifndef CUWR_MOTION_ESTIMATOR_H
#define CUWR_MOTION_ESTIMATOR_H

#include "cuwr_img.h"
#include "cuwr_imgdata_priv.h"
#include <vector>

#ifdef CUWR_WITH_QT
#include <QImage>
#include <QPainter>
#endif

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

    #ifdef CUWR_WITH_QT
        QImage toImage(const int blockSize=64) const{
            float maxLen = 0.0f;
            for (const auto & v : data_){
                const float len = v.length();
                if (len > maxLen)
                    maxLen = len;
            }
            QImage result(this->width()*blockSize,this->height()*blockSize,QImage::Format_RGB888);
            result.fill(Qt::white);
            QPainter painter(&result);
            painter.setPen(QPen(Qt::red,10));
            for (size_t r=0 ;r<this->height() ; ++r ){
                for (size_t c=0 ;c<this->width() ; ++c){
                    const cuwr_vec2 v = this->get(r,c);
                    QLineF line(0,0,v.x,v.y);
                    line.setLength( (blockSize/2)*(v.length()/maxLen) );
                    line.translate(QPointF( c*blockSize+blockSize/2, r*blockSize+blockSize/2 ));
                    painter.drawLine(line);
                }
            }
            return result;
        }
    #endif

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
        enum search_algorithm_t{
            Search_Exhaustive,
            Search_ThreeStep
        };

        MotionEstimator(size_t blockSize=16, int searchWindow=8)
            :blockSize_(blockSize > 0 ? blockSize : 16)
            ,searchWindow_(searchWindow > 0 ? searchWindow : 8)
        {
            cuwr::result_t err = module.load("cuwr_motion.ptx");
            if (err == cuwr::CUDA_SUCCESS_){
                this->calc_mad = module.function("cuwr_MAD",&err);
                this->three_step = module.function("cuwr_three_step_search",&err);
            }
            if (err != cuwr::CUDA_SUCCESS_)
                throw cuwr::Exception(err);
        }
        VectorField estimateMotionField(const cuwr::Image& image1,
                                        const cuwr::Image& image2,
                                        search_algorithm_t alg = Search_ThreeStep)
        {
            switch (alg){
            case Search_Exhaustive:
                return this->exhaustiveSearch(image1,image2);
            case Search_ThreeStep:
                return this->threeStepSearch(image1,image2);
            default:
                break;
            }
            return VectorField();
        }

    private:
        VectorField exhaustiveSearch(const cuwr::Image& image1,const cuwr::Image& image2)
        {
            VectorField result;
            if (image1.size() == image2.size()){
                result.resize( ((image1.width()+blockSize_)-1)/blockSize_,
                               ((image1.height()+blockSize_)-1)/blockSize_ );
                if (result.isEmpty()==false){
                    cuwr::DeviceArray<cuwr_mad_result_t, cuwr::DeviceMemPinnedAllocator> perBlockResult;
                    perBlockResult.resize(result.count(), cuwr_mad_result_t());
                    cuwr::DeviceValue<cuwr_dim2> off_dev = cuwr_dim2(0,0);
                    cuwr::KernelLaunchParams params;
                    params.autodetect(image1.size(),blockSize_);
                    params.setSharedMemoryCount(sizeof(float));
                    image1.pushData(params);
                    image1.pushHeader(params);
                    image2.pushData(params);
                    image2.pushHeader(params);
                    params.push(off_dev);
                    params.push(perBlockResult);

                    for (int i=-searchWindow_ ; i<=searchWindow_ ; ++i){
                        for (int j=-searchWindow_ ; j<=searchWindow_ ; ++j){
                            off_dev = cuwr_dim2(i,j);
                            if (cuwr::result_t r = cuwr::launch_kernel(this->calc_mad,params)){
                                throw Exception(r);
                            }
                            cuwr::cuStreamSynchronize(0);
                        }
                    }
                    const cuwr_mad_result_t * ptr = (const cuwr_mad_result_t*)perBlockResult.hostAddress();
                    const float maxMadValue = image1.bytesPerPixel()*255.0f;
                    for (size_t i=0 ; i<perBlockResult.count() ; ++i){
                        if (ptr[i].madValue > 0.0 && ptr[i].madValue < maxMadValue && (ptr[i].offset.x || ptr[i].offset.y)){
                            result.set(i,cuwr_vec2(ptr[i].offset.x,ptr[i].offset.y));
                         //  std::cout << "block moved: " << i << " " << ptr[i].madValue << ", "
                           //       << ptr[i].offset.x << "x" << ptr[i].offset.y << "\n";
                        }
                    }
                }
            }
            return result;
        }

        VectorField threeStepSearch(const cuwr::Image& image1,const cuwr::Image& image2)
        {
            VectorField result;
            if (image1.size() == image2.size()){
                result.resize( ((image1.width()+blockSize_)-1)/blockSize_,
                               ((image1.height()+blockSize_)-1)/blockSize_ );
                if (result.isEmpty()==false){

                    size_t searchStep = searchWindow_;
                    cuwr::DeviceArray<cuwr_mad_result_t, cuwr::DeviceMemPinnedAllocator> perBlockResult;
                    perBlockResult.resize(result.count(), cuwr_mad_result_t());
                    cuwr::DeviceArray<cuwr_dim2> off_dev;
                    off_dev.resize(result.count(),cuwr_dim2(0,0));
                    cuwr::KernelLaunchParams params;
                    params.autodetect(image1.size(),blockSize_);
                    params.setSharedMemoryCount(sizeof(float)*9);
                    image1.pushData(params);
                    image1.pushHeader(params);
                    image2.pushData(params);
                    image2.pushHeader(params);
                    params.push(&searchStep);
                    params.push(off_dev);
                    params.push(perBlockResult);

                    while (searchStep > 0){
                        if (cuwr::result_t r = cuwr::launch_kernel(this->three_step,params)){
                            throw Exception(r);
                        }
                        cuwr::cuStreamSynchronize(0);
                        searchStep /= 2;
                    }
                    const cuwr_mad_result_t * ptr = (const cuwr_mad_result_t*)perBlockResult.hostAddress();
                    const float maxMadValue = image1.bytesPerPixel()*255.0f;
                    for (size_t i=0 ; i<perBlockResult.count() ; ++i){
                        if (ptr[i].madValue > 0.0 && ptr[i].madValue < maxMadValue && (ptr[i].offset.x || ptr[i].offset.y)){
                            result.set(i,cuwr_vec2(ptr[i].offset.x,ptr[i].offset.y));
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
        cuwr::function_t three_step;
    };
}

#endif // CUWR_MOTION_ESTIMATOR_H

