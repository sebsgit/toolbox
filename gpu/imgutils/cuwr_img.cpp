#include "cuwr_img.h"

namespace cuwr{

    namespace priv{
        static cuwr::Module img_module;
        static cuwr::function_t swap_rgb;
        static cuwr::function_t set_pixels;
        static cuwr::function_t copy;
        static cuwr::result_t init_module(){
            if (img_module.isLoaded() == false){
                const cuwr::result_t err = img_module.load("cuwr_imgutils.ptx");
                if (err == cuwr::CUDA_SUCCESS_){
                    priv::swap_rgb = img_module.function("cuwr_swap_rgb");
                    priv::set_pixels = img_module.function("cuwr_set_pixels");
                    priv::copy = img_module.function("cuwr_copy");
                }
                return err;
            }
            return cuwr::CUDA_SUCCESS_;
        }
    }

    void Image::maxImageSize(const Gpu &gpu, size_t *maxWidth, size_t *maxHeight){
        *maxHeight = gpu.attribute(cuwr::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y_);
        *maxWidth = gpu.attribute(cuwr::CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X_);
    }

    static size_t get_bpp(const cuwr::image_format_t fmt){
        return (fmt==Format_Gray8 ? 1 : (fmt==Format_Rgb24 ? 3 : 4));
    }

    static cuwr_image_kernel_data_t empty_data(){
        cuwr_image_kernel_data_t result;
        result.bpp = 0;
        result.height = 0;
        result.width = 0;
        result.widthStep = 0;
        return result;
    }

    Image::Image()
        :format_(cuwr::Format_invalid)
    {
        this->header_ = empty_data();
        if (cuwr::result_t res = priv::init_module()){
            throw Exception(res);
        }
    }
    Image::Image(const size_t width, const size_t height, const image_format_t fmt)
        :Image(width,height,width*get_bpp(fmt),fmt)
    {
        if (cuwr::result_t res = priv::init_module()){
            throw Exception(res);
        }
    }

    Image::Image(const size_t width,
                 const size_t height,
                 const size_t widthStep,
                 const image_format_t fmt)
        :format_(fmt)
    {
        if (cuwr::result_t res = priv::init_module()){
            throw Exception(res);
        }
        if (width>0 && height>0 && fmt!=Format_invalid){
            cuwr_image_kernel_data_t tmp = empty_data();
            this->data_.resize(widthStep*height);
            tmp.bpp = get_bpp(fmt);
            tmp.height = height;
            tmp.width = width;
            tmp.widthStep = widthStep;
            this->header_ = tmp;
            this->recalculate_kernel_size();
        } else{
            this->header_ = empty_data();
        }
    }
    Image::Image(const Image &other)
        :header_(other.header_)
        ,data_(other.data_)
        ,format_(other.format_)
        ,offset_(0)
    {
        this->recalculate_kernel_size();
    }
    Image::Image(Image &&other)
        :header_(std::move(other.header_))
        ,data_(std::move(other.data_))
        ,format_(other.format_)
        ,offset_(0)
    {
        this->recalculate_kernel_size();
    }
    Image& Image::operator = (const Image& other){
        this->header_ = other.header_;
        this->data_ = other.data_;
        this->format_ = other.format_;
        this->offset_ = 0;
        this->recalculate_kernel_size();
        return *this;
    }
    Image& Image::operator = (Image&& other){
        this->header_ = std::move(other.header_);
        this->data_ = std::move(other.data_);
        this->format_ = other.format_;
        this->offset_ = 0;
        this->recalculate_kernel_size();
        return *this;
    }

    Image::~Image(){

    }
    size_t Image::byteCount() const{
        return this->data_.size();
    }
    size_t Image::height() const{
        const cuwr_image_kernel_data_t tmp = this->header_;
        return tmp.height;
    }
    size_t Image::width() const{
        const cuwr_image_kernel_data_t tmp = this->header_;
        return tmp.width;
    }
    void Image::setAutoSync(bool on, cuwr::stream_t stream){
        this->autoSync_ = on;
        this->autoSyncStream_ = stream;
    }

    void Image::swapRgb(){
        this->prepare_launch();
        if(result_t r = cuwr::launch_kernel(priv::swap_rgb,params_)){
            throw Exception(r);
        }
    }

    void Image::fill(const unsigned char r,
                     const unsigned char g,
                     const unsigned char b)
    {
        this->prepare_launch();
        params_.push(&r);
        params_.push(&g);
        params_.push(&b);
        if(result_t r = cuwr::launch_kernel(priv::set_pixels,params_)){
            throw Exception(r);
        }
    }

    /*
     finds the smallest 2d rectangle with area >= n_elems
     both width and height of the rectangle must be powers of 2
     */
    static void find_2d_box(const size_t n_elems, size_t * w, size_t * h){
        size_t p2 = 1;
        while (p2*p2 < n_elems){
            p2 *= 2;
        }
        *w = p2;
        size_t p2_h = p2/2;
        while (p2*p2_h > n_elems){
            p2_h /= 2;
        }
        if (p2_h == 0)
            p2_h = 1;
        *h = (p2_h*p2 > n_elems) ? p2_h :p2_h*2;
    }


    /* guess best grid dimensions for per-pixel processing of the image */
    static void guess_size(KernelLaunchParams * params,
                           const size_t imageWidth,
                           const size_t imageHeight)
    {
        const size_t blockWidth = 32;
        const size_t blockHeight = 32;
        const size_t threadsInBlock = blockWidth*blockHeight;
        const size_t nPixels = imageWidth*imageHeight;
        const size_t blocksNeeded = (nPixels/threadsInBlock)+1;
        size_t gridW, gridH;
        find_2d_box(blocksNeeded,&gridW,&gridH);
        params->setBlockSize(blockWidth,blockHeight);
        params->setGridSize(gridW,gridH);
    }

    Image Image::copy(size_t x, size_t y, size_t w, size_t h) const{
        x = std::min(x,this->width()-1);
        y = std::min(y,this->height()-1);
        w = std::min(w,this->width());
        h = std::min(h,this->height());
        Image result(w,h,this->format_);
        KernelLaunchParams params;
        guess_size(&params,result.width(),result.height());
        params.push(result.data_);
        params.push(result.header_);
        params.push(this->data_);
        params.push(this->header_);
        params.push(&x);
        params.push(&y);
        params.push(&w);
        params.push(&h);
        if (cuwr::result_t r = cuwr::launch_kernel(priv::copy,params)){
            throw Exception(r);
        }
        result.setAutoSync(this->autoSync_,this->autoSyncStream_);
        if (this->autoSync_)
            result.sync(this->autoSyncStream_);
        return result;
    }

    void Image::load(const unsigned char *data){
        this->data_.load(data);
    }

    void Image::recalculate_kernel_size(){
        guess_size(&params_,this->width(),this->height());
    }

    void Image::prepare_launch(){
        this->offset_ = 0;
        this->params_.clearParameters();
        this->params_.push(this->data_);
        this->params_.push(this->header_);
        this->params_.push(&this->offset_);
    }

    void Image::sync(stream_t stream) const{
        if (result_t r = cuwr::cuStreamSynchronize(stream))
            throw Exception(r);
    }

#ifdef CUWR_WITH_QT

    static QImage::Format toQFmt(const cuwr::image_format_t fmt){
        switch (fmt){
        case cuwr::Format_Gray8: return QImage::Format_Indexed8;
        case cuwr::Format_Rgb24: return QImage::Format_RGB888;
        case cuwr::Format_Rgba32: return QImage::Format_ARGB32;
        default: break;
        }
        return QImage::Format_Invalid;
    }
    static cuwr::image_format_t fromQFmt(const QImage::Format fmt){
        switch (fmt){
        case QImage::Format_Indexed8: return cuwr::Format_Gray8;
        case QImage::Format_RGB888: return cuwr::Format_Rgb24;
        case QImage::Format_ARGB32: return cuwr::Format_Rgba32;
        default: break;
        }
        return cuwr::Format_invalid;
    }

    QImage Image::toQImage() const{
        if (this->autoSync_)
            this->sync(this->autoSyncStream_);
        const cuwr_image_kernel_data_t tmp = this->header_;
        QImage image((const uchar *)data_.dataPtr().hostp_,tmp.width,tmp.height,tmp.widthStep,toQFmt(this->format_));
        return image.copy();
    }
    Image Image::fromQImage(const QImage& image){
        Image result;
        if (image.isNull()==false){
            cuwr::image_format_t fmt = fromQFmt(image.format());
            QImage source = image;
            if (fmt == cuwr::Format_invalid){
                source = image.convertToFormat(QImage::Format_RGB888);
                fmt = fromQFmt(source.format());
            }
            if (fmt != cuwr::Format_invalid){
                result = Image(source.width(),source.height(),source.bytesPerLine(),fmt);
                result.load(source.constBits());
            }
        }
        return result;
    }

#endif

}
