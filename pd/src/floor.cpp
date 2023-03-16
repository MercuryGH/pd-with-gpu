#include <primitive/floor.h>

namespace primitive
{
    void Floor::collision_handle(Eigen::Vector3f& pos) const 
    {
        if (pos.y() < y)
        {
            pos.y() = y;
        }
    }

    void Floor::generate_visualized_model(Eigen::MatrixXd& V, Eigen::MatrixXi& F) const
    {

    }
}