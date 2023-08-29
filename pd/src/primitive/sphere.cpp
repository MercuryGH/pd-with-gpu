#include <primitive/sphere.h>

#include <meshgen/mesh_generator.h>

namespace primitive
{
    bool Sphere::collision_handle(pd::SimVector3 &pos) const
    {
        const pd::SimScalar EPS = 0.05f;
        pd::SimVector3 diff = pos - center_point;
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

    pd::SimVector3 Sphere::center() const
    {
        return center_point;
    }

    void Sphere::set_center(pd::SimVector3 center)
    {
        center_point = center;
    }
}
