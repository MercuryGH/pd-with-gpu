#include <primitive/sphere.h>

#include <meshgen/mesh_generator.h>

namespace primitive
{
    bool Sphere::collision_handle(Eigen::Vector3f &pos) const
    {
        const float EPS = 0.05f;
        Eigen::Vector3f diff = pos - center_point;
        if (diff.norm() < radius + EPS)
        {
            diff.normalize();
            diff *= radius + EPS;
            pos = center_point + diff;
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
