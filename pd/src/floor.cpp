#include <primitive/floor.h>

namespace primitive
{
    bool Floor::collision_handle(Eigen::Vector3f& pos) const 
    {
        if (pos.y() < center_point.y())
        {
            pos.y() = center_point.y();
			return true;
        }
		return false;
    }

    void Floor::generate_visualized_model(Eigen::MatrixXd& V, Eigen::MatrixXi& F) const
    {
        std::vector<Eigen::RowVector3d> pos;
		std::vector<Eigen::RowVector3i> faces;

        constexpr int n_rows = 11;
        constexpr int n_cols = 11;

        for (int i = 0; i < n_rows; i++)
		{
			for (int j = 0; j < n_cols; j++)
			{
				const double x_offset = static_cast<double>(i - n_rows / 2);
				const double z_offset = static_cast<double>(j - n_cols / 2);
				// generate model first, then apply offset.
				pos.emplace_back(x_offset, 0, z_offset);

				if (i == n_rows - 1 || j == n_cols - 1)
				{
					continue;
				}

				const int lower_lcorner = i * n_cols + j;
				const int upper_lcorner = i * n_cols + j + 1;
				const int lower_rcorner = (i + 1) * n_cols + j;
				const int upper_rcorner = (i + 1) * n_cols + j + 1;

				faces.emplace_back(upper_rcorner, lower_lcorner, upper_lcorner);
				faces.emplace_back(lower_rcorner, lower_lcorner, upper_rcorner);
			}
		}

        V.resize(pos.size(), 3);
        F.resize(faces.size(), 3);
		for (int i = 0; i < pos.size(); i++)
		{
			V.row(i) = pos[i];
		}
		for (int i = 0; i < faces.size(); i++)
		{
			F.row(i) = faces[i];
		}
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