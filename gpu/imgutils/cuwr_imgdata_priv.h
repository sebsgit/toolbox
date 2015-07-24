#ifndef CUWR_IMGDATA_PRIV_H
#define CUWR_IMGDATA_PRIV_H

struct cuwr_image_kernel_data_t{
    size_t width;           /* width in pixels */
    size_t height;          /* height in pixels */
    size_t widthStep;       /* scan line size */
    size_t bpp;             /* bytes per pixel */
    cuwr_image_kernel_data_t(size_t w=0, size_t h=0,
                             size_t ws=0, size_t bp=0)
        :width(w)
        ,height(h)
        ,widthStep(ws)
        ,bpp(bp)
    {
    }
};

struct cuwr_dim2{
    int x;
    int y;
    cuwr_dim2(int ax=0, int ay=0)
        :x(ax)
        ,y(ay)
    {}
    bool operator == (const cuwr_dim2& other) const{
        return x==other.x && y==other.y;
    }
    bool operator != (const cuwr_dim2& other) const{
        return !(*this==other);
    }
};

struct cuwr_vec2{
    float x;
    float y;
    cuwr_vec2(float ax=0.0f, float ay=0.0f)
        :x(ax)
        ,y(ay)
    {}
};

#endif
