#include <primitive/block.h>

namespace primitive
{
    bool Block::collision_handle(pd::SimVector3& pos) const 
    {
        const auto is_minimum_3 = [](pd::SimScalar cmp, pd::SimScalar y, pd::SimScalar z)
        {
            return cmp <= y && cmp <= z;
        };

        constexpr static pd::SimScalar EPS = 0.05f;
        pd::SimVector3 tmp_half_xyz = half_xyz + pd::SimVector3(EPS, EPS, EPS);

        // AABB detection
        pd::SimVector3 min_xyz_point = center_point - tmp_half_xyz;
        pd::SimVector3 relative_pos = pos - min_xyz_point;
        pd::SimVector3 xyz = tmp_half_xyz * 2.f;

        // check if point in AABB
        if (relative_pos.x() > 0 && relative_pos.x() < xyz.x())
        {
            if (relative_pos.y() > 0 && relative_pos.y() < xyz.y())
            {
                if (relative_pos.z() > 0 && relative_pos.z() < xyz.z())
                {
                    // find the delta to the nearest point 
                    pd::SimVector3 delta_xyz;
                    for (int i = 0; i < 3; i++)
                    {
                        if (relative_pos[i] < xyz[i] - relative_pos[i])
                        {
                            delta_xyz[i] = -relative_pos[i];
                        }
                        else
                        {
                            delta_xyz[i] = xyz[i] - relative_pos[i];
                        }
                    }

                    // handle (project to the nearest plane)
                    if (is_minimum_3(std::abs(delta_xyz.x()), std::abs(delta_xyz.y()), std::abs(delta_xyz.z())))
                    {
                        pos.x() += delta_xyz.x();
                    }
                    else if (is_minimum_3(std::abs(delta_xyz.y()), std::abs(delta_xyz.x()), std::abs(delta_xyz.z())))
                    {
                        pos.y() += delta_xyz.y();
                    }
                    else if (is_minimum_3(std::abs(delta_xyz.z()), std::abs(delta_xyz.x()), std::abs(delta_xyz.y())))
                    {
                        pos.z() += delta_xyz.z();
                    }

                    return true;
                }
            }
        }
        return false;
    }

    void Block::generate_visualized_model(Eigen::MatrixXd& V, Eigen::MatrixXi& F) const
    {
        const auto incrementer = [this](bool inc_x, bool inc_y, bool inc_z)
        {
            // double x_val = inc_x ? center_point.x() + 0.5 * x : center_point.x() - 0.5 * x;
            // double y_val = inc_y ? center_point.y() + 0.5 * y : center_point.y() - 0.5 * y;
            // double z_val = inc_z ? center_point.z() + 0.5 * z : center_point.z() - 0.5 * z;

            double x_val = inc_x ? half_xyz.x() : -half_xyz.x();
            double y_val = inc_y ? half_xyz.y() : -half_xyz.y();
            double z_val = inc_z ? half_xyz.z() : -half_xyz.z();
            return Eigen::Vector3d(x_val, y_val, z_val);
        };

        V.resize(8, 3);
        F.resize(12, 3);

        //     7*-----*6
        //     /|    /|
        //    / |   / |
        //  4*-----*5 |
        //   | 3*--|--*2
        //   | /   | /
        //   |/    |/
        //  0*-----*1
        V.row(0) = incrementer(0, 0, 0);
        V.row(1) = incrementer(1, 0, 0);
        V.row(2) = incrementer(1, 1, 0);
        V.row(3) = incrementer(0, 1, 0);
        V.row(4) = incrementer(0, 0, 1);
        V.row(5) = incrementer(1, 0, 1);
        V.row(6) = incrementer(1, 1, 1);
        V.row(7) = incrementer(0, 1, 1);

        F.row(0) = Eigen::Vector3i(4, 0, 1);
        F.row(1) = Eigen::Vector3i(4, 1, 5);
        F.row(2) = Eigen::Vector3i(5, 1, 2);
        F.row(3) = Eigen::Vector3i(5, 2, 6);
        F.row(4) = Eigen::Vector3i(6, 2, 3);
        F.row(5) = Eigen::Vector3i(6, 3, 7);
        F.row(6) = Eigen::Vector3i(4, 3, 0);
        F.row(7) = Eigen::Vector3i(4, 7, 3);
        F.row(8) = Eigen::Vector3i(4, 5, 7);
        F.row(9) = Eigen::Vector3i(7, 5, 6);
        F.row(10) = Eigen::Vector3i(3, 0, 1);
        F.row(11) = Eigen::Vector3i(3, 1, 2);
    }

    pd::SimVector3 Block::center() const 
    {
        return center_point;
    }

    void Block::set_center(pd::SimVector3 center)
	{
        center_point = center;
	}
}