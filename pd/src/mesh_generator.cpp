#include <iostream>
#include <meshgen/mesh_generator.h>

#include <igl/boundary_facets.h>

namespace meshgen {
	std::pair<Eigen::MatrixXd, Eigen::MatrixXi> generate_plane(int x, int y, int usub, int vsub)
	{
		const int u_n_verts = usub + 1;
		const int v_n_verts = vsub + 1;
		const int n_verts = u_n_verts * v_n_verts;

		const int n_quads = usub * vsub;
		const int n_tris = n_quads * 2;

		const float du = 1.0f / (float)usub;
		const float dv = 1.0f / (float)vsub;

		Eigen::MatrixXd V(n_verts, 3);
		Eigen::MatrixXi F(n_tris, 3);

		int face_cnt = 0;
		float u = 0;
		for (int i = 0; i < u_n_verts; i++, u += du)
		{
			float v = 0;
			for (int j = 0; j < v_n_verts; j++, v += dv)
			{
				const float vertex_x = (u - 0.5f) * x;
				const float vertex_y = 0;
				const float vertex_z = (v - 0.5f) * y;

				const int idx = i * v_n_verts + j;
	
                V.row(idx) << (double)vertex_x, (double)vertex_y, (double)vertex_z;

				if (i < usub && j < vsub)
				{
                	F.row(face_cnt++) << idx, idx + 1, idx + v_n_verts + 1;
                	F.row(face_cnt++) << idx, idx + v_n_verts + 1, idx + v_n_verts;
				}
			}
		}

		return { V, F };
	}

	std::pair<Eigen::MatrixXd, Eigen::MatrixXi> generate_cloth(int n_rows, int n_cols)
	{
		return generate_plane(n_rows, n_cols, n_rows, n_cols);
	}

	std::pair<Eigen::MatrixXd, Eigen::MatrixXi> generate_sphere(float radius, int usub, int vsub, float urange, float vrange)
	{
		const bool closed_meridian = (urange == 1.0f); // generate circle with closed meridian
		const int modular = usub * (vsub + 1);
    
		const int u_n_verts = usub + 1;
		const int v_n_verts = vsub + 1;

		const int n_verts = u_n_verts * v_n_verts;
		const int n_quads = usub * vsub;
		const int n_tris = n_quads * 2;

		std::vector<Eigen::RowVector3d> vertices;
		std::vector<Eigen::RowVector3i> faces;

		const float du = urange / (float)(u_n_verts - 1);
		const float dv = vrange / (float)(v_n_verts - 1);

		const auto sphere_coord = [](float theta, float phi) {
			return Eigen::Vector3f(
				std::sin(theta) * std::sin(phi),
				std::cos(phi),
				std::cos(theta) * std::sin(phi)
			);
		};

		float u = 0;
		for (int i = 0; i < u_n_verts; i++, u += du)
		{
			const float theta = u * 2 * M_PI;
			float v = 0;
			for (int j = 0; j < v_n_verts; j++, v += dv)
			{
				const float phi = v * M_PI;
				Eigen::Vector3f pos = sphere_coord(theta, phi) * radius;

				const int idx = i * v_n_verts + j;
				if (closed_meridian == true && idx >= modular)
				{
					continue;
				}

				vertices.emplace_back(pos.transpose().cast<double>());
				if (i < usub && j < vsub)
				{
					const int v1 = idx;
					const int v2 = idx + 1;
					const int v3 = closed_meridian ? (idx + v_n_verts + 1) % modular : idx + v_n_verts + 1;
					const int v4 = closed_meridian ? (idx + v_n_verts) % modular : idx + v_n_verts;
                	faces.emplace_back(v1, v2, v3);
                	faces.emplace_back(v1, v3, v4);
				}
			}
		}

		Eigen::MatrixXd V(vertices.size(), 3);
		for (int i = 0; i < vertices.size(); i++)
		{
			V.row(i) = vertices.at(i);
		}
		Eigen::MatrixXi F(faces.size(), 3);
		for (int i = 0; i < faces.size(); i++)
		{
			F.row(i) = faces.at(i);
		}

		return { V, F };
	}

	std::pair<Eigen::MatrixXd, Eigen::MatrixXi> generate_hemisphere(float radius, int usub, int vsub)
	{
		return generate_sphere(radius, usub, vsub, 1.0f, 0.5f);
	}

	Eigen::Vector3f cylinder_coord(float theta, float height) 
	{
		return Eigen::Vector3f(
			std::sin(theta),
			height, 
			std::cos(theta)
		);
	}

	/**
	 * @brief 
	 * @todo BUG INCLUDED
	 * 
	 * @param radius 
	 * @param usub 
	 * @param cir_vsub 
	 * @param urange 
	 * @param height 
	 * @param vertex_n_offset 
	 * @return std::pair<Eigen::MatrixXd, Eigen::MatrixXi> 
	 */
	std::pair<Eigen::MatrixXd, Eigen::MatrixXi> generate_circle_plane(float radius, int usub, int cir_vsub, int urange, float height, int vertex_n_offset)
	{
		const int u_n_verts = usub + 1;
		const int v_n_verts = cir_vsub + 1;

		const int n_verts = u_n_verts * v_n_verts;
		const int n_quads = usub * cir_vsub;
		const int n_tris = n_quads * 2;

		const float du = urange / (float)(u_n_verts - 1);
		const float dv = 1.0 / (float)(v_n_verts - 1);

		Eigen::MatrixXd V(n_verts, 3);
		Eigen::MatrixXi F(n_tris, 3);

		float u = 0;
		int face_cnt = 0;
		for (int i = 0; i < u_n_verts; i++, u += du)
		{
			const float theta = u * 2 * M_PI;
			float v = 0;
			for (int j = 0; j < v_n_verts; j++, v += dv)
			{
				float r = v * radius;
				Eigen::Vector3f pos = cylinder_coord(theta, height) * radius;
				pos.y() = height;

				const int idx = i * v_n_verts + j;
				V.row(idx) << (double)pos.x(), (double)pos.y(), (double)pos.z(); 

				const int offset_idx = vertex_n_offset + idx;
				if (i < usub && j < cir_vsub)
				{
					const int v0 = offset_idx;
					const int v1 = offset_idx + v_n_verts;
					const int v2 = offset_idx + v_n_verts + 1;
					const int v3 = offset_idx + 1;

					F.row(face_cnt++) << v0, v2, v1;
					F.row(face_cnt++) << v0, v3, v2;
				}
			}
		}

		return { V, F };
	}

	std::pair<Eigen::MatrixXd, Eigen::MatrixXi> generate_cylinder(float radius, float height, int usub, int vsub, int capsub, float urange, float vrange)
	{
		const bool closed_meridian = (urange == 1.0f); // generate circle with closed meridian
		const int modular = usub * (vsub + 1);

		const int u_n_verts = usub + 1;
		const int v_n_verts = vsub + 1;

		const int n_body_verts = u_n_verts * v_n_verts;
		const int n_body_quads = usub * vsub;
		const int n_body_tris = n_body_quads * 2;

		std::vector<Eigen::RowVector3d> vertices;
		std::vector<Eigen::RowVector3i> faces;

		const float du = urange / (float)(u_n_verts - 1);
		const float dv = vrange / (float)(v_n_verts - 1);

		float u = 0;
		for (int i = 0; i < u_n_verts; i++, u += du)
		{
			const float theta = u * 2 * M_PI;
			float v = 0;
			for (int j = 0; j < v_n_verts; j++, v += dv)
			{
				const float vertex_y = (v - 0.5f) * height;
				Eigen::Vector3f pos = cylinder_coord(theta, vertex_y);
				pos = { pos.x() * radius, pos.y(), pos.z() * radius };

				const int idx = i * v_n_verts + j;
				if (closed_meridian == true && idx >= modular)
				{
					continue;
				}

				vertices.emplace_back(pos.transpose().cast<double>());
				if (i < usub && j < vsub)
				{
					const int v1 = idx;
					const int v2 = idx + 1;
					const int v3 = closed_meridian ? (idx + v_n_verts + 1) % modular : idx + v_n_verts + 1;
					const int v4 = closed_meridian ? (idx + v_n_verts) % modular : idx + v_n_verts;
                	faces.emplace_back(v1, v3, v2);
                	faces.emplace_back(v1, v4, v3);
				}
			}
		}

		if (capsub > 0)
		{
			vertices.clear();
			faces.clear();
			auto [V, F] = generate_circle_plane(radius, usub, capsub, urange, height / 2, vertices.size());
			for (int i = 0; i < V.rows(); i++)
			{
				Eigen::RowVector3d vrowi = V.row(i);
				vertices.emplace_back(vrowi);
			}
			for (int i = 0; i < F.rows(); i++)
			{
				Eigen::RowVector3i frowi = F.row(i);
				faces.emplace_back(frowi);
			}

			auto [V2, F2] = generate_circle_plane(radius, usub, capsub, urange, -height / 2, vertices.size());
			for (int i = 0; i < V2.rows(); i++)
			{
				vertices.push_back(V2.row(i));
			}
			for (int i = 0; i < F2.rows(); i++)
			{
				faces.push_back(F2.row(i));
			}
		}

		Eigen::MatrixXd V(vertices.size(), 3);
		for (int i = 0; i < vertices.size(); i++)
		{
			V.row(i) = vertices.at(i);
		}
		Eigen::MatrixXi F(faces.size(), 3);
		for (int i = 0; i < faces.size(); i++)
		{
			F.row(i) = faces.at(i);
		}

		return { V, F };
	}

	std::pair<Eigen::MatrixXd, Eigen::MatrixXi> generate_cone(float radius, float height, int usub, int vsub, int capsub, float urange, float vrange)
	{
		const int u_n_verts = usub + 1;
		const int v_n_verts = vsub + 1;

		const int n_body_verts = u_n_verts * v_n_verts;
		const int n_body_quads = usub * vsub;
		const int n_body_tris = n_body_quads * 2;

		std::vector<Eigen::RowVector3d> vertices;
		std::vector<Eigen::RowVector3i> faces;

		const float du = urange / (float)(u_n_verts - 1);
		const float dv = vrange / (float)(v_n_verts - 1);

		Eigen::MatrixXd V(vertices.size(), 3);
		for (int i = 0; i < vertices.size(); i++)
		{
			V.row(i) = vertices.at(i);
		}
		Eigen::MatrixXi F(faces.size(), 3);
		for (int i = 0; i < faces.size(); i++)
		{
			F.row(i) = faces.at(i);
		}

		return { V, F };
	}

	std::tuple<Eigen::MatrixXd, Eigen::MatrixXi, Eigen::MatrixXi> generate_bar(int x, int y, int z, int usub, int vsub)
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
					// TODO: This is problematic since it causes non-manifold in some cases
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