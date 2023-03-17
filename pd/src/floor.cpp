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
        std::vector<Eigen::RowVector3d> pos;
		std::vector<Eigen::RowVector3i> faces;

        constexpr int n_rows = 20;
        constexpr int n_cols = 20;

        for (int i = 0; i < n_rows; i++)
		{
			for (int j = 0; j < n_cols; j++)
			{
				const double x_offset = static_cast<double>(i);
				const double z_offset = static_cast<double>(j);
				pos.emplace_back(x_offset, this->y, z_offset);

				if (i == n_rows - 1 || j == n_cols - 1)
				{
					continue;
				}

				const int lower_lcorner = i * n_cols + j;
				const int upper_lcorner = i * n_cols + j + 1;
				const int lower_rcorner = (i + 1) * n_cols + j;
				const int upper_rcorner = (i + 1) * n_cols + j + 1;

				faces.emplace_back(lower_lcorner, upper_rcorner, upper_lcorner);
				faces.emplace_back(lower_lcorner, lower_rcorner, upper_rcorner);
			}
		}

        V.resize(pos.size(), 3);
        F.resize(faces.size(), 3);
		for (auto i = 0; i < pos.size(); i++)
		{
			V.row(i) = pos[i];
		}
		for (auto i = 0; i < faces.size(); i++)
		{
			F.row(i) = faces[i];
		}
    }

    Eigen::Vector3f Floor::center() const 
	{
		return Eigen::Vector3f(0, y, 0);
	}
	
	void Floor::set_center(Eigen::Vector3f center)
	{
		y = center.y();
	}
}