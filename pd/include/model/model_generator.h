#pragma once

#include <vector>
#include <Eigen/Core>

namespace generator {
	std::pair<Eigen::MatrixXd, Eigen::MatrixXi> generate_cloth(int n_rows, int n_cols)
	{
		if (n_rows < 1 || n_cols < 1)
		{
			printf("Error: Invalid arguments!");
		}

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

	std::tuple<Eigen::MatrixXd, Eigen::MatrixXi, Eigen::MatrixXi> generate_bar(int x, int y, int z)
	{
		if (x < 1 || y < 1 || z < 1)
		{
			printf("Error: Invalid arguments!");
		}
		// x, y, z <-> width, height, depth
		const auto base3_vertex_id = [x, y, z](int i, int j, int k) 
		{
			return i * y * z + j * z + 	k;
		};
		Eigen::MatrixXd V(x * y * z, 3);

		for (int i = 0; i < x; i++)
		{
			for (int j = 0; j < y; j++)
			{
				for (int k = 0; k < z; k++)
				{
					const auto cur_row = base3_vertex_id(i, j, k);
					V.row(cur_row) = Eigen::Vector3d((double)i, (double)j, (double)k);
				}
			}
		}

		const auto tet_cnt = (x - 1) * (y - 1) * (z - 1) * 5;

		// tetrahedron face, not triangle
		Eigen::MatrixXi T(tet_cnt, 4);
		for (int i = 0; i < x - 1; i++)
		{
			for (int j = 0; j < y - 1; j++)
			{
				for (int k = 0; k < z - 1; k++)
				{
					//     7*-----*6
					//     /|    /|
					//    / |   / |
					//  4*-----*5 |
					//   | 3*--|--*2
					//   | /   | /
					//   |/    |/
					//  0*-----*1
					const int v0 = base3_vertex_id(i, j, k);
					const int v1 = base3_vertex_id(i + 1, j, k);
					const int v2 = base3_vertex_id(i + 1, j + 1, k);
					const int v3 = base3_vertex_id(i, j + 1, k);
					const int v4 = base3_vertex_id(i, j, k + 1);
					const int v5 = base3_vertex_id(i + 1, j, k + 1);
					const int v6 = base3_vertex_id(i + 1, j + 1, k + 1);
					const int v7 = base3_vertex_id(i, j + 1, k + 1);

					const int cur_row = (i * (y - 1) * (z - 1) + j * (z - 1) + k) * 5;

					// create tetrahedron by splitting the cube (there are multiple splitting methods)
					if ((i + j + k) % 2 == 1)
					{
						T.row(cur_row) = Eigen::RowVector4i(v1, v0, v5, v2);
						T.row(cur_row + 1) = Eigen::RowVector4i(v5, v2, v7, v6);
						T.row(cur_row + 2) = Eigen::RowVector4i(v7, v0, v5, v4);
						T.row(cur_row + 3) = Eigen::RowVector4i(v2, v0, v7, v3);
						T.row(cur_row + 4) = Eigen::RowVector4i(v5, v0, v7, v2);
					}
					else
					{
						T.row(cur_row) = Eigen::RowVector4i(v3, v1, v4, v0);
						T.row(cur_row + 1) = Eigen::RowVector4i(v6, v1, v3, v2);
						T.row(cur_row + 2) = Eigen::RowVector4i(v4, v1, v6, v5);
						T.row(cur_row + 3) = Eigen::RowVector4i(v6, v3, v4, v7);
						T.row(cur_row + 4) = Eigen::RowVector4i(v3, v1, v6, v4);	
					}
				}
			}
		}

		// extract boundary facets for rendering only.
		// the simultaion process does not require the facet data
		Eigen::MatrixXi boundary_facets; 
    	igl::boundary_facets(T, boundary_facets); 

		// inverse face based
		T = T.rowwise().reverse().eval(); 
    	boundary_facets = boundary_facets.rowwise().reverse().eval();

		return { V, T, boundary_facets };
	}
}
