#ifndef MESHRENDERBUFFER_H
#define MESHRENDERBUFFER_H

#include <QOpenGLBuffer>
#include <QDebug>

#include "Mesh.h"

class MeshRenderBuffer
{
    template <typename Vec>
    static GLint byteCount(const Vec &v) noexcept
    {
        return static_cast<GLint>(sizeof(v[0]) * v.size());
    }
public:
    explicit MeshRenderBuffer(const Mesh &mesh)
        :mesh_ {mesh}
    {

    }

    void allocate(int width, int height)
    {

        if(!vertex_buffer_.create())
        {
            qDebug() << "vertex buffer create failed";
        }
        vertex_buffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);

        if(!texture_buffer_.create())
        {
            qDebug() << "texture buffer create failed";
        }
        texture_buffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);

        if (!index_buffer_.create())
        {
            qDebug() << "index buffer create failed";
        }
        index_buffer_.setUsagePattern(QOpenGLBuffer::StaticDraw);

        const auto vertices {mesh_.generateVertexCoords(width, height)};
        vertex_buffer_.bind();
        vertex_buffer_.allocate(vertices.data(), byteCount(vertices));

        const auto textureCoordinates {mesh_.generateTextureCoords()};
        texture_buffer_.bind();
        texture_buffer_.allocate(textureCoordinates.data(), byteCount(textureCoordinates));

        const auto indices {mesh_.generateIndices()};
        index_buffer_.bind();
        index_buffer_.allocate(indices.data(), byteCount(indices));
    }

    void update(int width, int height)
    {
        const auto vertices {mesh_.generateVertexCoords(width, height)};
        vertex_buffer_.bind();
        vertex_buffer_.write(0, vertices.data(), byteCount(vertices));

        const auto textureCoordinates {mesh_.generateTextureCoords()};
        texture_buffer_.bind();
        texture_buffer_.write(0, textureCoordinates.data(), byteCount(textureCoordinates));

        const auto indices {mesh_.generateIndices()};
        index_buffer_.bind();
        index_buffer_.write(0, indices.data(), byteCount(indices));
    }

    void bindVertexBuffer()
    {
        vertex_buffer_.bind();
    }

    void bindTextureBuffer()
    {
        texture_buffer_.bind();
    }

    void bindIndexBuffer()
    {
        index_buffer_.bind();
    }

    void release()
    {
        vertex_buffer_.release();
        texture_buffer_.release();
        index_buffer_.release();
    }

    GLuint indexCount() const
    {
        return mesh_.indexCount();
    }

private:
    const Mesh &mesh_;
    QOpenGLBuffer vertex_buffer_ {QOpenGLBuffer::VertexBuffer};
    QOpenGLBuffer texture_buffer_ {QOpenGLBuffer::VertexBuffer};
    QOpenGLBuffer index_buffer_ {QOpenGLBuffer::IndexBuffer};
};

#endif // MESHRENDERBUFFER_H
