#include <primitive/torus.h>

#include <meshgen/mesh_generator.h>

namespace primitive {
    bool Torus::collision_handle(Eigen::Vector3f &pos) const
    {
        const float EPS = 0.03;
        Eigen::Vector3f diff_xz = pos - center_point;
        diff_xz.y() = 0; // This only suits when the torus is placed horizontally
        Eigen::Vector3f tube_center = diff_xz.normalized() * main_radius + center_point;

        Eigen::Vector3f diff = pos - tube_center;

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

    Eigen::Vector3f Torus::center() const
    {
        return center_point;
    }

    void Torus::set_center(Eigen::Vector3f center)
    {
        center_point = center;
    }
}