#include <primitive/sphere.h>

#include <meshgen/mesh_generator.h>

namespace primitive
{
    bool Sphere::collision_handle(Eigen::Vector3f &pos) const
    {
        constexpr float EPS = 0.05f;
        if ((pos - center_point).norm() < radius + EPS)
        {
            Eigen::Vector3f dir = pos - center_point;
            dir.normalize();
            dir *= radius + EPS;
            pos = center_point + dir;
            return true;
        }
        return false;
    }

    void Sphere::generate_visualized_model(Eigen::MatrixXd &V, Eigen::MatrixXi &F) const
    {
        auto ret = meshgen::generate_sphere(radius);
        V = ret.first, F = ret.second;
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
