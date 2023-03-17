#pragma once

#include <vector>
#include <Eigen/Core>

namespace model {
	std::pair<Eigen::MatrixXd, Eigen::MatrixXi> generate_cloth(int n_rows, int n_cols)
	{
		std::vector<Eigen::RowVector3d> cloth_pos;
		std::vector<Eigen::RowVector3i> cloth_faces;

		for (int i = 0; i < n_rows; i++)
		{
			for (int j = 0; j < n_cols; j++)
			{
				const double x_offset = static_cast<double>(i);
				const double y_offset = static_cast<double>(j);
				cloth_pos.emplace_back(x_offset, y_offset, 0);

				if (i == n_rows - 1 || j == n_cols - 1)
				{
					continue;
				}

				const int lower_lcorner = i * n_cols + j;
				const int upper_lcorner = i * n_cols + j + 1;
				const int lower_rcorner = (i + 1) * n_cols + j;
				const int upper_rcorner = (i + 1) * n_cols + j + 1;

				cloth_faces.emplace_back(lower_lcorner, upper_rcorner, upper_lcorner);
				cloth_faces.emplace_back(lower_lcorner, lower_rcorner, upper_rcorner);
			}
		}

		Eigen::MatrixXd V(cloth_pos.size(), 3);
		Eigen::MatrixXi F(cloth_faces.size(), 3);
		for (auto i = 0; i < cloth_pos.size(); i++)
		{
			V.row(i) = cloth_pos[i];
		}
		for (auto i = 0; i < cloth_faces.size(); i++)
		{
			F.row(i) = cloth_faces[i];
		}
		return { V, F };
	}
}
