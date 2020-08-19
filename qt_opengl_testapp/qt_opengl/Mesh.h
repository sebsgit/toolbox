#ifndef MESH_H
#define MESH_H

#include <vector>
#include <QOpenGLFunctions>

class Mesh
{
public:
    virtual ~Mesh() noexcept = default;
    virtual std::vector<GLfloat> generateVertexCoords(int w, int h) const = 0;
    virtual std::vector<GLuint> generateIndices() const = 0;
    virtual GLint indexCount() const = 0;

    std::vector<GLfloat> generateTextureCoords() const
    {
        std::vector<GLfloat> result {generateTextureCoordsInternal()};
        if (texture_transform_)
        {
            for (size_t i = 0; i < result.size(); i += 2)
            {
                QPointF p {texture_transform_(
                    {static_cast<double>(result[i]), static_cast<double>(result[i + 1])}
                )};
                result[i] = static_cast<GLfloat>(p.x());
                result[i + 1] = static_cast<GLfloat>(p.y());
            }
        }
        return result;
    }

    void attachTextureTransform(std::function<QPointF(QPointF)> t)
    {
        texture_transform_ = std::move(t);
    }

protected:
    virtual std::vector<GLfloat> generateTextureCoordsInternal() const = 0;

protected:
    std::function<QPointF(QPointF)> texture_transform_;
};

class SquareMesh : public Mesh
{
public:
    std::vector<GLfloat> generateVertexCoords(int w, int h) const override
    {
        return {
            0.0f, h * 1.0f, 0.0f,
            w * 1.0f, h * 1.0f, 0.0f,
            0.0f, 0.0f, 0.0f,
            w * 1.0f, 0.0f, 0.0f
        };
    }

    std::vector<GLfloat> generateTextureCoordsInternal() const override
    {
        return {
            0.0f, 1.0f,
            1.0f, 1.0f,
            0.0f, 0.0f,
            1.0f, 0.0f,
        };
    }

    std::vector<GLuint> generateIndices() const override
    {
        return {
            0, 2, 1, 3
        };
    }

    GLint indexCount() const override
    {
        return 4;
    }
};

#endif // MESH_H
