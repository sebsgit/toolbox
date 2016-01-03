#ifndef CUWR_MOTION_ESTIMATOR_H
#define CUWR_MOTION_ESTIMATOR_H

#include "cuwr_img.h"
#include "cuwr_imgdata_priv.h"
#include <vector>

#ifdef CUWR_WITH_QT
class QImage;
#endif

namespace cuwr{
    class VectorField{
    public:
        VectorField(const size_t width=0, const size_t height=0);
        ~VectorField();
        void resize(const size_t x, const size_t y, const cuwr_vec2& initValue=cuwr_vec2(0,0));
        void resize(const cuwr_dim2& dim);
        bool isEmpty() const;
        size_t count() const;
        size_t width() const;
        size_t height() const;
        cuwr_dim2 size() const;
        std::vector<cuwr_vec2> to_vector() const;
        void set(size_t index,const cuwr_vec2& value);
        cuwr_vec2 get(size_t row, size_t col) const;

    #ifdef CUWR_WITH_QT
        QImage toImage(const int blockSize=64) const;
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
    class MotionEstimator : public ContextEntity{
    public:
        enum search_algorithm_t{
            Search_Exhaustive,
            Search_ThreeStep
        };

        MotionEstimator(size_t blockSize=16, int searchWindow=8);
        VectorField estimateMotionField(const cuwr::Image& image1,
                                        const cuwr::Image& image2,
                                        search_algorithm_t alg = Search_ThreeStep) const;

    private:
        /* for each block, match every possible block in the search window */
        VectorField exhaustiveSearch(const cuwr::Image& image1,const cuwr::Image& image2) const;
        /* for each block, match 8 blocks on the diagonal and x,y axes with step S,
            then pick the one with smallest MAD value, set S=S/2 and repeat the search
            procedure for the found block */
        VectorField threeStepSearch(const cuwr::Image& image1,const cuwr::Image& image2) const;

    private:
        size_t blockSize_;
        int searchWindow_;
        cuwr::Module module;
        cuwr::function_t calc_mad;
        cuwr::function_t three_step;
    };
}

#endif // CUWR_MOTION_ESTIMATOR_H
