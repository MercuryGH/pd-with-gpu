#include <primitive/sphere.h>

namespace primitive
{
    bool Sphere::collision_handle(Eigen::Vector3f &pos) const
    {
        if ((pos - center_point).norm() < radius)
        {
            Eigen::Vector3f dir = pos - center_point;
            dir.normalize();
            dir *= radius;
            pos = center_point + dir;
            return true;
        }
        return false;
    }

    void Sphere::generate_visualized_model(Eigen::MatrixXd &V, Eigen::MatrixXi &F) const
    {
        constexpr int resolution = 20;

        V.resize(resolution * resolution, 3);
        F.resize(2 * (resolution - 1) * resolution, 3);

        // create sphere vertices using sphere coordinates
        for (int i = 0; i < resolution; i++)
        {
            float z = radius * std::cos(PI * (float)i / (float(resolution - 1)));
            for (int j = 0; j < resolution; j++)
            {
                const float sin_val = std::sin(PI * (float)i / (float(resolution - 1)));
                float x = radius * sin_val * std::cos(2 * PI * (float)j / (float)resolution);
                float y = radius * sin_val * std::sin(2 * PI * (float)j / (float)resolution);

                // apply center offset
                // V.row(i * resolution + j) << (double)(x + center_point.x()), (double)(y + center_point.y()), (double)(z + center_point.z());
                V.row(i * resolution + j) << (double)x, (double)y, (double)z;
            }
        }

        // assign sphere triangle
        for (int i = 0; i < resolution - 1; i++)
        {
            for (int j = 0; j < resolution; j++)
            {
                const int v1 = i * resolution + j;
                const int v2 = (i + 1) * resolution + j;
                const int v3 = (i + 1) * resolution + (j + 1) % resolution;
                const int v4 = i * resolution + (j + 1) % resolution;

                F.row(2 * (resolution * i + j)) << v1, v2, v3;
                F.row(2 * (resolution * i + j) + 1) << v4, v1, v3;
            }
        }
    }

    Eigen::Vector3f Sphere::center() const
    {
        return center_point;
    }

    void Sphere::set_center(Eigen::Vector3f center)
    {
        center_point = center;
    }
}
