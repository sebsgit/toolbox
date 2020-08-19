#ifndef CPUGRID_H
#define CPUGRID_H

#include <vector>
#include <functional>

namespace cpu
{
    struct Point
    {
        int x = 0;
        int y = 0;

        Point operator- (const Point &p) const
        {
            return Point{x - p.x, y - p.y};
        }

        Point operator+ (const Point &p) const
        {
            return Point{ x + p.x, y + p.y };
        }
    };

    struct Size
    {
        int width = 0;
        int height = 0;
    };

    class Grid
    {
    public:
        explicit Grid(Size node_count, Size input_image_size, Size target_image_size)
        {
            node_count_ = node_count;
            target_image_size_ = target_image_size;
            input_image_size_ = input_image_size;

            const int step_x = input_image_size.width / (node_count.width - 1);
            const int step_y = input_image_size.height / (node_count.height - 1);

            for (int y = 0; y < node_count.height; ++y)
            {
                for (int x = 0; x < node_count.width; ++x)
                {
                    Point node;
                    node.x = x * step_x;
                    node.y = y * step_y;
                    coords_.push_back(node);
                }
            }
        }

        int stepX() const
        {
            return target_image_size_.width / (node_count_.width - 1);
        }

        int stepY() const
        {
            return target_image_size_.height / (node_count_.height - 1);
        }

        std::vector<Point> coords_;
        Size node_count_;
        Size target_image_size_;
        Size input_image_size_;
    };

    class Image
    {
    public:

        unsigned char * data = nullptr;
        Size size;

        template <typename T = unsigned char>
        T get_pixel(int w, int h) const
        {
            if (w >= size.width || h >= size.height || w < 0 || h < 0)
            {
                return T{};
            }
            return static_cast<T>(data[w + size.width * h]);
        }

        void set_pixel(int w, int h, unsigned char p)
        {
            if (w >= size.width || h >= size.height || w < 0 || h < 0)
            {
                return;
            }
            data[w + size.width * h] = p;
        }

        void draw_point(Point p)
        {
            const int size = 2;
            for (int x = -size; x <= size; ++x)
            {
                for (int y = -size; y <= size ; ++y)
                {
                    set_pixel(p.x + x, p.y + y, 255);
                }
            }
        }

    };

    class LUT
    {
    public:
        std::vector<Point> coords;
    };

    class Matrix
    {
    public:
        float a = 1.0;
        float b = 0.0;
        float c = 0.0;
        float d = 1.0;

        float det() const
        {
            return a * d - b * c;
        }

        Point transform(Point p) const
        {
            return Point{static_cast<int>(a * p.x + b * p.y),
                static_cast<int>(c * p.x + d * p.y)};
        }

        Matrix inverse() const
        {
            const float dx{1.0f / det()};
            return Matrix{ dx * d, - dx * b, -dx * c, dx * a };
        }

        static std::pair<Matrix, Point> solve(const Point X, const Point Y, const Point Z,
            const Point P, const Point Q, const Point R)
        {
            Matrix m;
            Point B;

            if (Y.y == X.y)
            {
                m.a = 1.0 * (Q.x - P.x) / (Y.x - X.x);
                m.c = 1.0 * (Q.y - P.y) / (Y.x - X.x);
                m.b = 1.0 * (R.x - P.x - m.a * (Z.x - X.x)) / (Z.y - X.y);
                m.d = 1.0 * (R.y - P.y - m.c * (Z.x - X.x)) / (Z.y - X.y);
            }
            else if (Z.x == X.x)
            {
                m.b = 1.0 * (R.x - P.x) / (Z.y - X.y);
                m.d = 1.0 * (R.y - P.y) / (Z.y - X.y);
                m.a = 1.0 * (Q.x - P.x - m.b * (Y.y - X.y)) / (Y.x - X.x);
                m.c = 1.0 * (Q.y - P.y - m.d * (Y.y - X.y)) / (Y.x - X.x);
            }
            else if (Y.x == X.x)
            {
                m.b = 1.0 * (Q.x - P.x) / (Y.y - X.y);
                m.d = 1.0 * (Q.y - P.y) / (Y.y - X.y);
                m.a = 1.0 * (R.x - P.x - m.b * (Z.y - X.y)) / (Z.x - X.x);
                m.c = 1.0 * (R.y - P.y - m.d * (Z.y - X.y)) / (Z.x - X.x);
            }
            else
            {
                throw "failed to solve";
            }

            B.x = P.x - m.a * X.x - m.b * X.y;
            B.y = P.y - m.c * X.x - m.d * X.y;

            return std::make_pair(m, B);
        }
    };

    static Grid inverseGrid(const Grid &input)
    {
        Grid result = input;
        result.target_image_size_ = input.input_image_size_;
        result.input_image_size_ = input.target_image_size_;
        for (auto & n : result.coords_)
        {
            n = Point{0, 0};
        }

        for (int y = 0; y < input.node_count_.height - 1; ++y)
        {
            for (int x = 0; x < input.node_count_.width - 1; ++x)
            {
                Point top_left = input.coords_[x + y * input.node_count_.width];
                Point top_right = input.coords_[x + 1 + y * input.node_count_.width];
                Point bottom_left = input.coords_[x + (y + 1) * input.node_count_.width];
                Point bottom_right = input.coords_[(x + 1) + (y + 1) * input.node_count_.width];

                Point target_tl{ x * result.stepX(),  y * result.stepY() };
                Point target_tr{ (x + 1) * result.stepX(),  y * result.stepY() };
                Point target_bl{ x * result.stepX(),  (y + 1) * result.stepY() };
                Point target_br{ (x + 1) * result.stepX(),  (y + 1) * result.stepY() };

                Point &top_left_inv = result.coords_[x + y * input.node_count_.width];
                Point &top_right_inv = result.coords_[x + 1 + y * input.node_count_.width];
                Point &bottom_left_inv = result.coords_[x + (y + 1) * input.node_count_.width];
                Point &bottom_right_inv = result.coords_[(x + 1) + (y + 1) * input.node_count_.width];

                auto upper_transform = Matrix::solve(
                    target_tl, target_tr, target_br,
                    top_left, top_right, bottom_right
                );
                upper_transform.first = upper_transform.first.inverse();

                auto lower_transform = Matrix::solve(
                    target_tl, target_bl, target_br,
                    top_left, bottom_left, bottom_right
                );
                lower_transform.first = lower_transform.first.inverse();

                top_left_inv = upper_transform.first.transform(target_tl - upper_transform.second);
                top_right_inv = upper_transform.first.transform(target_tr - upper_transform.second);
                bottom_left_inv = lower_transform.first.transform(target_bl - lower_transform.second);
                bottom_right_inv = lower_transform.first.transform(target_br - lower_transform.second);
            }
        }

        return result;
    }

    static void apply(const std::function<void(Point, Point)> sampler,
        const Point &a, const Point &b, const Point &c, const Point &d,
        const Point &target_a, const Point &target_b, const Point &target_c, const Point &target_d
        )
    {
        const int steps_x = std::abs(target_a.x - target_b.x);
        const int steps_y = std::abs(target_a.y - target_c.y);

        for (int y = 0; y < steps_y; ++y)
        {
            for (int x = 0; x < steps_x; ++x)
            {
                const int out_x = x + target_a.x;
                const int out_y = y + target_a.y;

                const float ratio_x = 1.0f * x / (steps_x - 1.0f);
                const float ratio_y = 1.0f * y / (steps_y - 1.0f);

                Point upper_line;
                upper_line.x = (1.0f - ratio_x) * a.x + ratio_x * b.x;
                upper_line.y = (1.0f - ratio_x) * a.y + ratio_x * b.y;

                Point lower_line;
                lower_line.x = (1.0f - ratio_x) * c.x + ratio_x * d.x;
                lower_line.y = (1.0f - ratio_x) * c.y + ratio_x * d.y;

                Point sample_loc;
                sample_loc.x = (1.0f - ratio_y) * upper_line.x + ratio_y * lower_line.x;
                sample_loc.y = (1.0f - ratio_y) * upper_line.y + ratio_y * lower_line.y;


                sampler(sample_loc, { out_x, out_y });

            }
        }
    }

    static void apply(const Image &in, Image &out,
        const Point &a, const Point &b, const Point &c, const Point &d,
        const Point &target_a, const Point &target_b, const Point &target_c, const Point &target_d)
    {
        auto sampler = [&in, &out](Point from, Point to)
        {
            out.set_pixel(to.x, to.y, in.get_pixel(from.x, from.y));
        };
        apply(sampler, a, b, c, d, target_a, target_b, target_c, target_d);
    }

    static LUT generateLUT(const Grid &grid, const Grid &original_grid)
    {
        LUT result;
        result.coords.resize(grid.target_image_size_.width * grid.target_image_size_.height);

        for (int y = 0; y < grid.node_count_.height - 1; ++y)
        {
            for (int x = 0; x < grid.node_count_.width - 1; ++x)
            {
                Point top_left = grid.coords_[x + y * grid.node_count_.width];
                Point top_right = grid.coords_[x + 1 + y * grid.node_count_.width];
                Point bottom_left = grid.coords_[x + (y + 1) * grid.node_count_.width];
                Point bottom_right = grid.coords_[(x + 1) + (y + 1) * grid.node_count_.width];

                Point target_tl{ x * original_grid.stepX(),  y * original_grid.stepY() };
                Point target_tr{ (x + 1) * original_grid.stepX(),  y * original_grid.stepY() };
                Point target_bl{ x * original_grid.stepX(),  (y + 1) * original_grid.stepY() };
                Point target_br{ (x + 1) * original_grid.stepX(),  (y + 1) * original_grid.stepY() };

                auto sampler = [&result, w = grid.target_image_size_.width](Point from, Point to)
                {
                    result.coords[to.x + to.y * w] = from;
                };

                apply(sampler,
                      top_left, top_right, bottom_left, bottom_right,
                      target_tl, target_tr, target_bl, target_br);
            }
        }

        return result;
    }

    //TODO: original grid not needed, only the patch step
    static void apply(const Image &in, Image &out, const Grid &grid, const Grid &original_grid)
    {
        for (int y = 0; y < grid.node_count_.height - 1; ++y)
        {
            for (int x = 0; x < grid.node_count_.width - 1; ++x)
            {
                Point top_left = grid.coords_[x + y * grid.node_count_.width];
                Point top_right = grid.coords_[x + 1 + y * grid.node_count_.width];
                Point bottom_left = grid.coords_[x + (y + 1) * grid.node_count_.width];
                Point bottom_right = grid.coords_[(x + 1) + (y + 1) * grid.node_count_.width];

                Point target_tl{ x * original_grid.stepX(),  y * original_grid.stepY() };
                Point target_tr{ (x + 1) * original_grid.stepX(),  y * original_grid.stepY() };
                Point target_bl{ x * original_grid.stepX(),  (y + 1) * original_grid.stepY() };
                Point target_br{ (x + 1) * original_grid.stepX(),  (y + 1) * original_grid.stepY() };

                apply(in, out, top_left, top_right, bottom_left, bottom_right,
                    target_tl, target_tr, target_bl, target_br);
            }
        }
    }

    static void apply(const Image &in, Image &out, const LUT &lut)
    {
        for (int y=0 ; y < out.size.height; ++y)
        {
            for (int x=0 ; x < out.size.width; ++x)
            {
                const auto p {lut.coords[x + y * out.size.width]};
                out.set_pixel(x, y, in.get_pixel(p.x, p.y));
            }
        }
    }

    static void overlay(Image &out, const Grid &original_grid)
    {
        for (int y = 0; y < original_grid.node_count_.height - 1; ++y)
        {
            for (int x = 0; x < original_grid.node_count_.width - 1; ++x)
            {
                auto top_left = original_grid.coords_[x + y * original_grid.node_count_.width];
                auto top_right = original_grid.coords_[x + 1 + y * original_grid.node_count_.width];
                auto bottom_left = original_grid.coords_[x + (y + 1) * original_grid.node_count_.width];
                auto bottom_right = original_grid.coords_[(x + 1) + (y + 1) * original_grid.node_count_.width];
                out.draw_point(top_left);
                out.draw_point(top_right);
                out.draw_point(bottom_left);
                out.draw_point(bottom_right);
            }
        }
    }
};

#endif // CPUGRID_H
