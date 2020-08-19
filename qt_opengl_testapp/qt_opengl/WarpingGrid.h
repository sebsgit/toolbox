#ifndef WARPINGGRID_H
#define WARPINGGRID_H

#include "Mesh.h"
#include <QDebug>
#include <QSize>

#include "CpuGrid.h"

class WarpingGrid : public Mesh
{
public:
    explicit WarpingGrid()
    {
        setNodeCount(2, 2);
    }

    std::vector<GLfloat> generateVertexCoords(int w, int h) const override
    {
        const float step_w {1.0f * w / (node_count_width_ - 1)};
        const float step_h {1.0f * h / (node_count_height_ - 1)};
        std::vector<GLfloat> result;
        for (int y = node_count_height_ - 1 ; y >= 0; --y)
        {
            for (int x = 0; x < node_count_width_; ++x)
            {
                const GLfloat node_x {x * step_w};
                const GLfloat node_y {y * step_h};
                const GLfloat node_z {0.0f};
                result.push_back(node_x);
                result.push_back(node_y);
                result.push_back(node_z);
            }
        }
        return result;
    }

    std::vector<GLfloat> generateTextureCoordsInternal() const override
    {
        return tex_coord_buffer_;
    }

    std::vector<GLuint> generateIndices() const override
    {
        return index_buffer_;
    }

    GLint indexCount() const override
    {
        return static_cast<GLint>(index_buffer_.size());
    }

    void setNodeCount(int w, int h)
    {
        node_count_width_ = qBound(2, w, 64);
        node_count_height_ = qBound(2, h, 64);
        updateBuffer();
    }

    cpu::Grid generateCpuWarpingGrid(QSize input_image_size, QSize target_image_size) const
    {
        cpu::Size node_count;
        node_count.width = node_count_width_;
        node_count.height = node_count_height_;

        cpu::Size inp_img_size {input_image_size.width(), input_image_size.height()};
        cpu::Size target_img_size {target_image_size.width(), target_image_size.height()};
        cpu::Grid result {node_count, inp_img_size, target_img_size};

        const auto tex_coords {generateTextureCoordsInternal()};
        for (size_t i = 0; i < result.coords_.size(); ++i)
        {
            result.coords_[i].x = tex_coords[2 * i] * input_image_size.width();
            result.coords_[i].y = (1.0 - tex_coords[2 * i + 1]) * input_image_size.height();
        }
        if (texture_transform_)
        {
            for (auto &coord : result.coords_)
            {
                QPointF r {texture_transform_(QPointF(1.0 * coord.x / input_image_size.width(), 1.0 * coord.y / input_image_size.height()))};
                coord.x = r.x() * input_image_size.width();
                coord.y = r.y() * input_image_size.height();
            }
        }
        return result;
    }

protected:
    void updateBuffer()
    {
        tex_coord_buffer_.clear();
        index_buffer_.clear();

        // generate texture buffer
        const float step_w {1.0f / (node_count_width_ - 1)};
        const float step_h {1.0f / (node_count_height_ - 1)};
        for (int y = node_count_height_ - 1 ; y >= 0; --y)
        {
            for (int x = 0; x < node_count_width_; ++x)
            {
                const GLfloat node_x {x * step_w};
                const GLfloat node_y {y * step_h};
                tex_coord_buffer_.push_back(node_x);
                tex_coord_buffer_.push_back(node_y);
            }
        }

        // triangle strip index buffer
        const GLuint lastPatchY {static_cast<GLuint>(node_count_height_ - 2)};
        for (GLuint y = 0; y <= lastPatchY ; y += 1)
        {
            for (GLuint x = 0; x <= static_cast<GLuint>(node_count_width_ - 1); ++x)
            {
                const GLuint top {y * (node_count_width_) + x};
                const GLuint bottom {(y + 1) * (node_count_width_) + x};
                index_buffer_.push_back(top);
                index_buffer_.push_back(bottom);
            }

            // degenerate triangle to get to the next row
            if (y != lastPatchY)
            {
                index_buffer_.push_back(index_buffer_.back());
                const GLuint top_left {(y + 1) * (node_count_width_)};
                index_buffer_.push_back(top_left);
            }
        }
    }

private:
    int node_count_width_ {0};
    int node_count_height_ {0};

    std::vector<GLfloat> tex_coord_buffer_;
    std::vector<GLuint> index_buffer_;
};

#endif // WARPINGGRID_H
