#include <primitive/sphere.h>

namespace primitive
{
    void Sphere::collision_handle(Eigen::Vector3f& pos) const 
    {

    }

    void Sphere::generate_visualized_model(Eigen::MatrixXd& V, Eigen::MatrixXi& F) const
    {

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