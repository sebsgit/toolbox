#ifndef AVW_PIXEL_FORMAT_HPP
#define AVW_PIXEL_FORMAT_HPP

extern "C" {
#include "libavutil/pixfmt.h"
}

namespace avw {
enum class pixel_format {
    none = AV_PIX_FMT_NONE,
    rgb24 = AV_PIX_FMT_RGB24,
    yuv420p = AV_PIX_FMT_YUV420P
    //...
    ,
    unknown
};

enum pixel_format from_av_format(int code);
AVPixelFormat to_av_format(enum pixel_format format);
const char* pixel_format_string(const enum pixel_format fmt);

} // namespace avw

#ifdef AVW_MEDIA_IMPL
enum avw::pixel_format avw::from_av_format(int code)
{
    if (code == AV_PIX_FMT_NONE)
        return avw::pixel_format::none;
    else if (code == AV_PIX_FMT_RGB24)
        return avw::pixel_format::rgb24;
    else if (code == AV_PIX_FMT_YUV420P)
        return avw::pixel_format::yuv420p;
    return avw::pixel_format::unknown;
}

AVPixelFormat avw::to_av_format(enum avw::pixel_format format)
{
    return static_cast<AVPixelFormat>(format);
}

#define MAKE_PXFMT_CASE(code)     \
    case avw::pixel_format::code: \
        return #code

const char* avw::pixel_format_string(const enum pixel_format fmt)
{
    switch (fmt) {
        MAKE_PXFMT_CASE(none);
        MAKE_PXFMT_CASE(rgb24);
        MAKE_PXFMT_CASE(yuv420p);
    default:
        return "none";
    }
}
#undef MAKE_PXFMT_CASE
#endif

#endif // AVW_PIXEL_FORMAT_HPP
