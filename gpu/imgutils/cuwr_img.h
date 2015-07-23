#ifndef CUWRIMAGEUTILS_H
#define CUWRIMAGEUTILS_H

#include "cuwrap.h"
#include "cuwr_imgdata_priv.h"

#ifdef CUWR_WITH_QT
#include <QImage>
#endif

namespace cuwr{
	enum image_format_t{
        Format_Gray8,
        Format_Rgb24,
        Format_Rgba32,
        Format_invalid
	};
	
	class Image{
	public:
        Image();
		Image(const size_t width, const size_t height, const image_format_t fmt);
        Image(const size_t width, const size_t height, const size_t widthStep, const image_format_t fmt);
        Image(Image&& other);
        Image& operator=(Image&& other);
        Image(const Image& other);
        Image& operator = (const Image& other);
		~Image();
        size_t byteCount() const;
        size_t width() const;
        size_t height() const;
        cuwr::image_format_t format() const{ return this->format_; }
        void swapRgb();
        void fill(const unsigned char r,
                  const unsigned char g,
                  const unsigned char b);
        void load(const unsigned char * data);

        void sync(cuwr::stream_t stream = 0);
    #ifdef CUWR_WITH_QT
        QImage toQImage() const;
        static Image fromQImage(const QImage& image);
    #endif

    private:
        void recalculate_kernel_size();
        void prepare_launch();
	private:
        cuwr::DeviceValue<cuwr_image_kernel_data_t, cuwr::DeviceMemPinnedAllocator> header_;
        cuwr::DeviceArray<unsigned char, cuwr::DeviceMemPinnedAllocator> data_;
        image_format_t format_;
        size_t offset_;
        cuwr::KernelLaunchParams params_;
	};
}

#endif
