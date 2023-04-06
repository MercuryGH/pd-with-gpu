#include <primitive/floor.h>

#include <meshgen/mesh_generator.h>

namespace primitive
{
    bool Floor::collision_handle(Eigen::Vector3f& pos) const 
    {
		constexpr static float EPS = 0.01;
        if (pos.y() < center_point.y() + EPS)
        {
            pos.y() = center_point.y() + EPS;
			return true;
        }
		return false;
    }

    void Floor::generate_visualized_model(Eigen::MatrixXd& V, Eigen::MatrixXi& F) const
    {
		auto ret = meshgen::generate_plane(11, 11, 1, 1);
		V = ret.first, F = ret.second;
    }

    Eigen::Vector3f Floor::center() const 
	{
		return center_point;
	}
	
	void Floor::set_center(Eigen::Vector3f center)
	{
		center_point.y() = center.y();
	}
}