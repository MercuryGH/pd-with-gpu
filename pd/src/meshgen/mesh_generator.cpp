#include <iostream>
#include <meshgen/mesh_generator.h>

#include <igl/boundary_facets.h>

namespace meshgen {
	std::pair<Eigen::MatrixXd, Eigen::MatrixXi> generate_plane(int x, int y, int usub, int vsub, Eigen::MatrixXd& UV)
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
		UV.resize(n_verts, 2);

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
				UV.row(idx) << (double)u, (double)v;

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
		Eigen::MatrixXd uv;
		return generate_plane(n_rows, n_cols, n_rows, n_cols, uv);
	}

	std::pair<Eigen::MatrixXd, Eigen::MatrixXi> generate_cloth(int n_rows, int n_cols, Eigen::MatrixXd& uv)
	{
		return generate_plane(n_rows, n_cols, n_rows, n_cols, uv);
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
	 * @brief used to generate bottom and top of a cylinder
	 * @note The bottom and top is not connected
	 */
	std::pair<Eigen::MatrixXd, Eigen::MatrixXi> generate_circle_plane(float radius, int usub, int cir_vsub, int urange, float height, int vertex_n_offset, bool inverse_normal)
	{
		printf("cir_vsub = %d\n", cir_vsub);
		const int u_n_verts = usub + 1;
		const int v_n_verts = cir_vsub + 1;

		const int n_verts = u_n_verts * v_n_verts;
		const int n_quads = usub * cir_vsub;
		const int n_tris = n_quads * 2;

		const float du = urange / (float)(u_n_verts - 1);
		const float dv = 1.0f / (float)(v_n_verts - 1);

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
				Eigen::Vector3f pos = cylinder_coord(theta, height);
				pos.x() *= r, pos.z() *= r;
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

					if (inverse_normal == false)
					{
						F.row(face_cnt++) << v0, v2, v1;
						F.row(face_cnt++) << v0, v3, v2;
					}
					else
					{
						F.row(face_cnt++) << v0, v1, v2;
						F.row(face_cnt++) << v0, v2, v3;
					}
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
			// bottom
			auto [V2, F2] = generate_circle_plane(radius, usub, capsub, urange, -height / 2, vertices.size(), true);
			for (int i = 0; i < V2.rows(); i++)
			{
				vertices.push_back(V2.row(i));
			}
			for (int i = 0; i < F2.rows(); i++)
			{
				faces.push_back(F2.row(i));
			}

			// top
			auto [V, F] = generate_circle_plane(radius, usub, capsub, urange, height / 2, vertices.size(), false);
			for (int i = 0; i < V.rows(); i++)
			{
				vertices.push_back(V.row(i));
			}
			for (int i = 0; i < F.rows(); i++)
			{
				faces.push_back(F.row(i));
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

	Eigen::Vector3f cone_coord(float theta, float y, float height) 
	{
		float scale = 1.0 - y / height;
		return Eigen::Vector3f(
			std::sin(theta) * scale,
			y, 
			std::cos(theta) * scale
		);
	}

	std::pair<Eigen::MatrixXd, Eigen::MatrixXi> generate_cone(float radius, float height, int usub, int vsub, int capsub, float urange, float vrange)
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
				const float vertex_y = v * height;
				Eigen::Vector3f pos = cone_coord(theta, vertex_y, height);
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
			// bottom
			auto [V2, F2] = generate_circle_plane(radius, usub, capsub, urange, 0, vertices.size(), true);
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

	Eigen::Vector3f torus_coord(float theta, float phi, float mr, float rr)
	{
		theta = -theta;

		const float rx = -std::cos(phi) * rr + mr;
		const float ry = std::sin(phi) * rr;
		const float rz = 0.0;

		const float x = rx * std::sin(theta) + rz * std::cos(theta);
		const float y = ry;
		const float z = -rx * std::cos(theta) + rz * std::sin(theta);

		return Eigen::Vector3f(x, y, z);
	}

	std::pair<Eigen::MatrixXd, Eigen::MatrixXi> generate_torus(float main_radius, float ring_radius, int usub, int vsub, float urange, float vrange)
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

		float u = 0;
		for (int i = 0; i < u_n_verts; i++, u += du)
		{
			const float theta = u * 2 * M_PI;
			float v = 0;
			for (int j = 0; j < v_n_verts; j++, v += dv)
			{
				const float phi = v * 2 * M_PI;
				
				Eigen::Vector3f pos = torus_coord(theta, phi, main_radius, ring_radius);
				Eigen::Vector3f center = torus_coord(theta, phi, main_radius, 0);

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

	std::tuple<Eigen::MatrixXd, Eigen::MatrixXi, Eigen::MatrixXi> generate_bar(int width, int height, int depth)
	{
		const int x_n_vertex = width + 1;
		const int y_n_vertex = height + 1;
		const int z_n_vertex = depth + 1;

		// x, y, z <-> width, height, depth
		const auto base3_vertex_id = [x_n_vertex, y_n_vertex, z_n_vertex](int i, int j, int k) 
		{
			return i * y_n_vertex * z_n_vertex + j * z_n_vertex + k;
		};
		Eigen::MatrixXd V(x_n_vertex * y_n_vertex * z_n_vertex, 3);

		for (int i = 0; i < x_n_vertex; i++)
		{
			for (int j = 0; j < y_n_vertex; j++)
			{
				for (int k = 0; k < z_n_vertex; k++)
				{
					const auto cur_row = base3_vertex_id(i, j, k);
					V.row(cur_row) = Eigen::Vector3d((double)i, (double)j, (double)k);
				}
			}
		}

		const auto tet_cnt = (x_n_vertex - 1) * (y_n_vertex - 1) * (z_n_vertex - 1) * 5;

		// tetrahedron face, not triangle
		Eigen::MatrixXi T(tet_cnt, 4);
		for (int i = 0; i < x_n_vertex - 1; i++)
		{
			for (int j = 0; j < y_n_vertex - 1; j++)
			{
				for (int k = 0; k < z_n_vertex - 1; k++)
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

					const int cur_row = (i * (y_n_vertex - 1) * (z_n_vertex - 1) + j * (z_n_vertex - 1) + k) * 5;

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