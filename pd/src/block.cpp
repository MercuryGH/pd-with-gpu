#include <primitive/block.h>

namespace primitive
{
    void Block::collision_handle(Eigen::Vector3f& pos) const 
    {

    }

    void Block::generate_visualized_model(Eigen::MatrixXd& V, Eigen::MatrixXi& F) const
    {

    }

    Eigen::Vector3f Block::center() const 
    {
        return center_point;
    }

    void Block::set_center(Eigen::Vector3f center)
	{
        center_point = center;
	}
}