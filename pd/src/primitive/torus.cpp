#include <primitive/torus.h>

#include <meshgen/mesh_generator.h>

namespace primitive {
    bool Torus::collision_handle(pd::SimVector3 &pos) const
    {
        const pd::SimScalar EPS = 0.03;
        pd::SimVector3 diff_xz = pos - center_point;
        diff_xz.y() = 0; // This only suits when the torus is placed horizontally
        pd::SimVector3 tube_center = diff_xz.normalized() * main_radius + center_point;

        pd::SimVector3 diff = pos - tube_center;

        if (diff.norm() < ring_radius + EPS)
        {
            diff.normalize();
            diff *= ring_radius + EPS;
            pos = tube_center + diff;
            return true;
        }
        return false;
    }

    void Torus::generate_visualized_model(Eigen::MatrixXd &V, Eigen::MatrixXi &F) const
    {
        auto ret = meshgen::generate_torus(main_radius, ring_radius);
        V = ret.first, F = ret.second;
    }

    pd::SimVector3 Torus::center() const
    {
        return center_point;
    }

    void Torus::set_center(pd::SimVector3 center)
    {
        center_point = center;
    }
}