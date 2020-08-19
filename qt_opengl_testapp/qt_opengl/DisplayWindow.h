#ifndef DISPLAYWINDOW_H
#define DISPLAYWINDOW_H

#include <QOpenGLShaderProgram>
#include <QOpenGLWidget>
#include <QOpenGLTexture>
#include <QOpenGLExtraFunctions>
#include <QOpenGLVertexArrayObject>
#include <QOpenGLBuffer>
#include <QOpenGLFramebufferObject>
#include <QElapsedTimer>
#include <QTransform>

#include "MeshRenderBuffer.h"
#include "WarpingGrid.h"
#include "DistortionParameters.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/calib3d.hpp>

namespace transforms
{
static QPointF rotateClockwise(QPointF p, qreal degrees)
{
    QTransform t;
    t.translate(0.5, 0.5);
    t.rotate(-degrees);
    t.translate(-0.5, -0.5);
    return t.map(p);
}

static QPointF undistort(QPointF p, DistortionParameters dist, int w, int h)
{
    QPointF t {(p.x() * w - dist.cx) / dist.fx, (p.y() * h - dist.cy) / dist.fy};
    const qreal r2 {t.x() * t.x() + t.y() * t.y()};
    const qreal r4 {r2 * r2};
    const qreal r6 {r4 * r2};
    const qreal factor {1.0 + r2 * dist.k1 + r4 * dist.k2 + r6 * dist.k3};

    t.rx() *= factor;
    t.ry() *= factor;
    t.rx() = t.x() * dist.fx + dist.cx;
    t.ry() = t.y() * dist.fy + dist.cy;
    return QPointF(t.x() / w, t.y() / h);
}
}

class DisplayWindow : public QOpenGLWidget {
    Q_OBJECT
public:
    DisplayWindow(QWidget* parent = nullptr)
        : QOpenGLWidget(parent)
    {
        mesh_buffer_.reset(new MeshRenderBuffer(mesh_));
        quad_buffer_.reset(new MeshRenderBuffer(fullscreen_quad_mesh_));

        fullscreen_quad_mesh_.attachTextureTransform([](QPointF coord) {
            return QPointF(coord.x(), 1.0 - coord.y());
        });
    }

    ~DisplayWindow() override
    {
        makeCurrent();
        texture_.reset();
        fbo_.reset();
        mesh_buffer_.reset();
        quad_buffer_.reset();
        doneCurrent();
    }

signals:
    void cpuImageRendered(QImage);
    void opencvImageRendered(QImage);

public slots:
    void loadImage(const QImage &image)
    {
        current_image_ = image;

        makeCurrent();
        texture_ = std::make_unique<QOpenGLTexture>(image, QOpenGLTexture::MipMapGeneration::DontGenerateMipMaps);
        texture_->setWrapMode(QOpenGLTexture::WrapMode::ClampToBorder);
        fbo_ = std::make_unique<QOpenGLFramebufferObject>(image.size());

        mesh_.attachTextureTransform([this](QPointF coord)
        {
            return transforms::undistort(
                    transforms::rotateClockwise(coord, rotation_),
                    distortion_,
                    current_image_.width(),
                    current_image_.height()
                    );
        });

        mesh_buffer_->allocate(image.width(), image.height());
        mesh_buffer_->update(image.width(), image.height());
        doneCurrent();

        updateCpuImage();
    }

    void setRotation(qreal angle)
    {
        rotation_ = angle;

        updateMesh();
    }

    void setDistortion(DistortionParameters p)
    {
        distortion_ = p;
        distortion_.cx = p.cx * current_image_.width();
        distortion_.cy = p.cy * current_image_.height();

        updateMesh();
    }

    void setDisplayGrid(bool enabled)
    {
        display_grid_ = enabled;

        updateCpuImage();
    }

    void setGridSize(int x, int y)
    {
        mesh_.setNodeCount(x, y);
        mesh_buffer_.reset(new MeshRenderBuffer(mesh_));
        makeCurrent();
        mesh_buffer_->allocate(current_image_.width(), current_image_.height());
        doneCurrent();
        updateMesh();
    }

protected:
    void initializeGL() override
    {
        glEnable(GL_TEXTURE_2D);
        glDisable(GL_DEPTH_TEST | GL_BLEND);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);

        const GLfloat borderColor[] = { 1.0f, 1.0f, 0.0f, 1.0f };
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, borderColor);

        shader_.addShaderFromSourceCode(QOpenGLShader::Vertex,
            "#version 430\n"
            "attribute highp vec4 vertex;\n"
            "attribute highp vec2 tex_coord;\n"
            "uniform highp mat4 matrix;\n"
            "out varying vec2 tex_coord_out;\n"
            "void main(void)\n"
            "{\n"
            "   tex_coord_out = tex_coord;\n"
            "   gl_Position = matrix * vertex;\n"
            "}");

        shader_.addShaderFromSourceCode(QOpenGLShader::Fragment,
            "#version 430\n"
            "uniform sampler2D tex;\n"
            "in vec2 tex_coord_out;\n"
            "void main(void)\n"
            "{\n"
            "   gl_FragColor = texture2D(tex, tex_coord_out);\n"
            "}");

        if (!shader_.link()) {
            qDebug() << "Program link error: " << shader_.log();
        }

        quad_buffer_->allocate(width(), height());

        gl_.initializeOpenGLFunctions();
    }

    void resizeGL(int w, int h) override
    {
        glViewport(0, 0, w, h);

        quad_buffer_->update(width(), height());
    }

    void paintGL() override
    {
        if (!texture_)
        {
            return;
        }

        fbo_->bind();
        renderTexture(texture_->textureId(), fbo_->size(), *mesh_buffer_);
        fbo_->release();

        renderTexture(fbo_->texture(), size(), *quad_buffer_);
    }

    void renderTexture(GLuint tex_id, QSize target_size, MeshRenderBuffer &mesh_buffer)
    {
        shader_.bind();

        int vertexLocation = shader_.attributeLocation("vertex");
        int textureArrayLoc = shader_.attributeLocation("tex_coord");
        int matrixLocation = shader_.uniformLocation("matrix");
        int textureLocation = shader_.uniformLocation("tex");

        gl_.glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex_id);

        QMatrix4x4 viewMat;
        viewMat.ortho(QRect(0, 0, target_size.width(), target_size.height()));
        glViewport(0, 0, target_size.width(), target_size.height());
        shader_.setUniformValue(matrixLocation, viewMat);
        shader_.setUniformValue(textureLocation, GL_TEXTURE0);

        mesh_buffer.bindVertexBuffer();
        shader_.enableAttributeArray(vertexLocation);
        shader_.setAttributeBuffer(vertexLocation, GL_FLOAT, 0, 3);

        mesh_buffer.bindTextureBuffer();
        shader_.enableAttributeArray(textureArrayLoc);
        shader_.setAttributeBuffer(textureArrayLoc, GL_FLOAT, 0, 2);

        mesh_buffer.bindIndexBuffer();
        glDrawElements(GL_TRIANGLE_STRIP, mesh_buffer.indexCount(), GL_UNSIGNED_INT, nullptr);

        shader_.disableAttributeArray(vertexLocation);
        shader_.disableAttributeArray(textureArrayLoc);

        mesh_buffer.release();
        shader_.release();
    }

    void updateCpuImage()
    {
        auto grid_test = mesh_.generateCpuWarpingGrid(current_image_.size(), current_image_.size());
        auto grid_inv = cpu::inverseGrid(grid_test);

        auto lut = cpu::generateLUT(grid_test, grid_inv);

        QImage gray = current_image_.convertToFormat(QImage::Format_Grayscale8);
        QImage result_grid_img(current_image_.size(), QImage::Format_Grayscale8);
        cpu::Image cpu_img {gray.bits(), cpu::Size{current_image_.width(), current_image_.height()}};
        cpu::Image output_img {result_grid_img.bits(), cpu::Size{result_grid_img.width(), result_grid_img.height()}};

        QElapsedTimer t;
        t.start();
        const auto then {std::chrono::high_resolution_clock::now()};

        cpu::apply(cpu_img, output_img, lut);
        const auto now {std::chrono::high_resolution_clock::now()};

        qDebug() << "LUT (unoptimized): " << t.elapsed() << "ms. / "
                 << std::chrono::duration_cast<std::chrono::microseconds>(now - then).count() << "microsec.";

        if (display_grid_)
            cpu::overlay(output_img, grid_inv);

        emit cpuImageRendered(result_grid_img);
    }

    void updateMesh()
    {
        makeCurrent();
        mesh_buffer_->update(current_image_.width(), current_image_.height());
        doneCurrent();
        update();
        updateCpuImage();
        processWithOpenCV();
    }

    void processWithOpenCV()
    {
        QImage gray = current_image_.convertToFormat(QImage::Format_Grayscale8);
        cv::Mat image(gray.height(), gray.width(), CV_8UC1, gray.bits());
        cv::Mat result(gray.height(), gray.width(), CV_8UC1);
        cv::Mat undist(gray.height(), gray.width(), CV_8UC1);

        cv::Mat camera = cv::Mat::eye(3, 3, CV_32FC1);
        camera.at<float>(0, 0) = distortion_.fx;
        camera.at<float>(1, 1) = distortion_.fy;
        camera.at<float>(0, 2) = distortion_.cx;
        camera.at<float>(1, 2) = distortion_.cy;

        cv::Mat dist_coeff = cv::Mat::zeros(1, 8, CV_32FC1);
        dist_coeff.at<float>(0, 0) = distortion_.k1;
        dist_coeff.at<float>(0, 1) = distortion_.k2;
        dist_coeff.at<float>(0, 4) = distortion_.k3;

        auto t = cv::getRotationMatrix2D(cv::Point2f{image.cols / 2.0f, image.rows / 2.0f}, -rotation_, 1.0);

        QElapsedTimer tm;
        tm.start();
        const auto then {std::chrono::high_resolution_clock::now()};

        cv::undistort(image, undist, camera, dist_coeff);
        cv::warpAffine(undist, result, t, result.size());

        const auto now {std::chrono::high_resolution_clock::now()};

        qDebug() << "OpenCV: " << tm.elapsed() << "ms. / "
                 << std::chrono::duration_cast<std::chrono::microseconds>(now - then).count() << "microsec.";

        emit opencvImageRendered(QImage(result.data, result.cols, result.rows, QImage::Format_Grayscale8));
    }

private:
    QOpenGLExtraFunctions gl_;

    QOpenGLShaderProgram shader_;
    std::unique_ptr<QOpenGLTexture> texture_;
    std::unique_ptr<QOpenGLFramebufferObject> fbo_;

    WarpingGrid mesh_;
    SquareMesh fullscreen_quad_mesh_;

    std::unique_ptr<MeshRenderBuffer> mesh_buffer_;
    std::unique_ptr<MeshRenderBuffer> quad_buffer_;

    QImage current_image_;

    qreal rotation_ {0.0};
    DistortionParameters distortion_;
    bool display_grid_ {true};
};

#endif // DISPLAYWINDOW_H
