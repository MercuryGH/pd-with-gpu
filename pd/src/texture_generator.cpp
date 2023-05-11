#include <texturegen/texture_generator.h>

namespace texturegen {
    void faded_checkerboard_texture(const int s, const int f,
                         Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& X,
                         Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic>& A)
    {
        X.resize(s * f, s * f);
        A.resize(s * f, s * f);
        for (int i = 0; i < s * f; i++)
        {
            const double x = double(i) / double(s * f - 1) * 2 - 1;
            for (int j = 0; j < s * f; j++)
            {
                const int u = i / f;
                const int v = j / f;
                const double y = double(j) / double(s * f - 1) * 2 - 1;
                const double r1 = std::min(std::max((1.0 - sqrt(x * x + y * y)) * 1.0, 0.0), 1.0);
                // const double r1 = std::min(std::max((1.0 - sqrt(x * x + y * y)) * 2.0, 0.0), 1.0);
                const double r3 = std::min(std::max((1.0 - sqrt(x * x + y * y)) * 3.0, 0.0), 1.0);
                // const double a = 3*r*r - 2*r*r*r;
                const auto smooth_step = [](const double w)
                {
                    return ((w * (w * 6.0 - 15.0) + 10.0) * w * w * w);
                };
                double a3 = smooth_step(r1);
                double a1 = smooth_step(r1);
                X(i, j) = (0.75 + 0.25 * a1) * (u % 2 == v % 2 ? 240 : 180);
                A(i, j) = a3 * 255;
            }
        }
    }
}