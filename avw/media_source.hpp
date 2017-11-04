#ifndef AVW_MEDIA_SOURCE_HPP
#define AVW_MEDIA_SOURCE_HPP

#include "pixel_format.hpp"

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/imgutils.h"
#include "libswscale/swscale.h"
}

#include <chrono>
#include <memory>
#include <string>

namespace avw {

enum class media_status {
    no_media,
    ready,
    invalid,
    unknown
};

enum class error {
    none = 0,
    eof,
    backend_error,
    unknown
};

void init();

class media_source;

class media_frame {
public:
    media_frame()
        : _frame(av_frame_alloc())
    {
    }
    media_frame(media_frame&& other)
        : _frame(other._frame)
        , _own_data(other._own_data)
    {
        other._frame = nullptr;
        other._own_data = false;
    }
    media_frame& operator=(media_frame&& other)
    {
        if (this->_own_data)
            av_freep(&this->_frame->data[0]);
        if (this->_frame)
            av_frame_free(&this->_frame);
        this->_frame = other._frame;
        this->_own_data = other._own_data;
        other._frame = nullptr;
        other._own_data = false;
        return *this;
    }
    ~media_frame()
    {
        if (this->_own_data)
            av_freep(&this->_frame->data[0]);
        if (this->_frame)
            av_frame_free(&this->_frame);
    }
    media_frame allocate(int width, int height, avw::pixel_format format, int alignment = 4) const
    {
        media_frame result;
        result._own_data = true;
        result.handle()->width = width;
        result.handle()->height = height;
        result.handle()->format = to_av_format(format);
        av_frame_copy_props(result.handle(), this->handle());
        av_image_alloc(result.handle()->data, result.handle()->linesize, width, height, to_av_format(format), alignment);
        return result;
    }
    int width() const { return this->_frame->width; }
    int height() const { return this->_frame->height; }
    enum pixel_format pixel_format() const { return from_av_format(this->backend_format()); }
    AVFrame* handle() { return this->_frame; }
    const AVFrame* handle() const { return this->_frame; }
    int backend_format() const { return this->_frame->format; }

    int64_t presentation_timestamp() const { return av_frame_get_best_effort_timestamp(this->_frame); }
    template <typename type = uint8_t>
    typename std::add_const<type>::type* data(size_t plane) const
    {
        return static_cast<typename std::add_const<type>::type*>(this->_frame->data[plane]);
    }

private:
    AVFrame* _frame;
    bool _own_data = false;

private:
    media_frame(const media_frame&) = delete;
    media_frame& operator=(const media_frame&) = delete;
};

class context_stream {
public:
    enum type {
        audio = AVMEDIA_TYPE_AUDIO,
        video = AVMEDIA_TYPE_VIDEO
    };

public:
    explicit context_stream(const media_source& media, type codec_type);
    virtual ~context_stream()
    {
        if (this->_context) {
            avcodec_free_context(&this->_context);
        }
    }
    enum type type() const { return this->_type; }
    int index() const { return this->_stream_index; }
    int backend_error() const { return this->_error; }
    const char* codec_name() const { return this->_codec ? this->_codec->name : "<invalid codec>"; }

    media_frame decode(AVPacket* packet)
    {
        media_frame frame;
        this->_error = avcodec_send_packet(this->_context, packet);
        if (this->_error == 0) {
            this->_error = avcodec_receive_frame(this->_context, frame.handle());
        }
        return frame;
    }

    double time_base() const
    {
        return this->_time_base;
    }

    void flush()
    {
        avcodec_flush_buffers(this->_context);
    }

protected:
    const media_source* _parent = nullptr;
    AVCodec* _codec = nullptr;
    AVCodecContext* _context = nullptr;
    int _stream_index = -1;
    double _time_base = 0.0f;
    int _error = 0;
    enum type _type;

private:
    context_stream(const context_stream&) = delete;
    context_stream& operator=(const context_stream&) = delete;
};

class video_stream : public context_stream {
public:
    explicit video_stream(const media_source& media)
        : context_stream(media, context_stream::video)
    {
    }

    int width() const
    {
        return this->_context ? this->_context->width : -1;
    }
    int height() const
    {
        return this->_context ? this->_context->height : -1;
    }
    enum pixel_format format() const
    {
        return from_av_format(this->_context ? this->_context->pix_fmt : -1);
    }
};

class image_scaler {
public:
    struct format {
        int width;
        int height;
        AVPixelFormat pix_fmt;
    };
    image_scaler(const format& input, const format& output)
    {
        this->_scaler = sws_getContext(input.width, input.height, input.pix_fmt,
            output.width, output.height, output.pix_fmt,
            SWS_BILINEAR, nullptr, nullptr, nullptr);
    }
    ~image_scaler()
    {
        sws_freeContext(this->_scaler);
    }
    bool rescale(const media_frame& input, media_frame& output)
    {
        return sws_scale(this->_scaler, input.handle()->data, input.handle()->linesize, 0, input.height(),
                   output.handle()->data, output.handle()->linesize)
            != 0;
    }

private:
    SwsContext* _scaler = nullptr;
};

class media_source {
public:
    media_source() = default;
    explicit media_source(const std::string& path)
    {
        this->set_media(path);
    }
    ~media_source()
    {
        if (this->_formatCtx)
            avformat_close_input(&this->_formatCtx);
    }
    std::string path() const { return this->_path; }
    enum error error() const { return from_backend_error(this->_backend_error); }
    int backend_error() const { return this->_backend_error; }
    const AVFormatContext* format_context() const { return this->_formatCtx; }

    void set_requested_pixel_format(enum pixel_format format)
    {
        this->_requested_format = format;
    }

    void set_media(const std::string& path)
    {
        this->_path.clear();
        this->_scaler.reset();
        this->_video_stream.reset();
        this->_backend_error = avformat_open_input(&this->_formatCtx, path.data(), nullptr, nullptr);
        if (this->_backend_error == 0) {
            this->_backend_error = avformat_find_stream_info(this->_formatCtx, nullptr);
            if (this->_backend_error >= 0) {
                this->_video_stream.reset(new video_stream(*this));
                this->_backend_error = this->_video_stream->backend_error();
                if (this->_backend_error == 0) {
                    this->_path = path;
                    if (this->_requested_format != pixel_format::none && this->_video_stream->format() != this->_requested_format) {
                        const int w = this->_video_stream->width();
                        const int h = this->_video_stream->height();
                        this->_scaler.reset(new image_scaler(
                            { w, h, static_cast<AVPixelFormat>(this->_video_stream->format()) },
                            { w, h, static_cast<AVPixelFormat>(this->_requested_format) }));
                    }
                }
            }
        }
    }

    const video_stream* video() const
    {
        return this->_video_stream.get();
    }

    media_frame next_frame()
    {
        if (this->_video_stream) {
            AVPacket packet;
            this->_backend_error = av_read_frame(this->_formatCtx, &packet);
            while (this->_backend_error == 0) {
                if (packet.stream_index == this->_video_stream->index()) {
                    media_frame frame = this->_video_stream->decode(&packet);
                    this->_backend_error = this->_video_stream->backend_error();
                    if (this->_backend_error == 0) {
                        if (this->_scaler) {
                            media_frame rescaled = frame.allocate(this->_video_stream->width(), this->_video_stream->height(), this->_requested_format);
                            if (this->_scaler->rescale(frame, rescaled))
                                frame = std::move(rescaled);
                        }
                        return frame;
                    }
                }
                this->_backend_error = av_read_frame(this->_formatCtx, &packet);
            }
        }
        return media_frame();
    }
    template <typename Base = std::chrono::milliseconds>
    auto duration() const
    {
        return std::chrono::duration_cast<Base>(std::chrono::microseconds(this->_formatCtx->duration));
    }
    void seek(double fraction)
    {
        fraction = fraction < 0 ? 0.0 : fraction > 1.0 ? 1.0 : fraction;
        const auto max_timestamp = this->_formatCtx->duration;
        const auto nearest_timestamp = static_cast<decltype(max_timestamp)>(max_timestamp * fraction);
        this->_backend_error = av_seek_frame(this->_formatCtx, -1, nearest_timestamp, AVSEEK_FLAG_ANY | AVSEEK_FLAG_BACKWARD);
        this->_video_stream->flush();
    }

private:
    static enum error from_backend_error(int code)
    {
        if (code == AVERROR_EOF)
            return avw::error::eof;
        else if (code == AVERROR_UNKNOWN)
            return avw::error::unknown;
        else if (code == 0)
            return avw::error::none;
        return avw::error::backend_error;
    }

private:
    AVFormatContext* _formatCtx = nullptr;
    int _backend_error = 0;
    enum pixel_format _requested_format = pixel_format::none;
    std::unique_ptr<video_stream> _video_stream;
    std::unique_ptr<image_scaler> _scaler;
    std::string _path;
};
} // namespace avw

#ifdef AVW_MEDIA_IMPL

void avw::init()
{
    avcodec_register_all();
    av_register_all();
}

avw::context_stream::context_stream(const media_source& media, enum type codec_type)
    : _parent(&media)
    , _type(codec_type)
{
    AVCodecParameters* params = nullptr;
    for (uint32_t i = 0; i < media.format_context()->nb_streams; ++i) {
        if (media.format_context()->streams[i]->codecpar->codec_type == static_cast<int>(codec_type)) {
            params = media.format_context()->streams[i]->codecpar;
            this->_stream_index = i;
            this->_time_base = av_q2d(media.format_context()->streams[i]->time_base);
            break;
        }
    }
    if (params && this->_stream_index >= 0) {
        this->_codec = avcodec_find_decoder(params->codec_id);
        if (this->_codec) {
            this->_context = avcodec_alloc_context3(this->_codec);
            this->_error = avcodec_parameters_to_context(this->_context, params);
            if (this->_error >= 0)
                this->_error = avcodec_open2(this->_context, this->_codec, nullptr);
        }
    }
}
#endif // AVW_MEDIA_IMPL

#endif // AVW_MEDIA_SOURCE_HPP
