#include <pd/deformable_mesh.h>
#include <pd/positional_constraint.h>
#include <pd/edge_strain_constraint.h>
#include <pd/bending_constraint.h>
#include <pd/tet_strain_constraint.h>

namespace pd
{
	void DeformableMesh::toggle_vertices_fixed(const std::unordered_set<int> &v, float wc)
	{
		for (const int vi : v)
		{
			if (is_vertex_fixed(vi) == false)
			{
				add_positional_constraint(vi, wc);
				fixed_vertices.insert(vi);
			}
			else
			{
				fixed_vertices.erase(vi);
			}
		}
	}

	void DeformableMesh::add_positional_constraint(int vi, float wc)
	{
		constraints.push_back(new PositionalConstraint(
			wc, vi, p
		));
	}

	void DeformableMesh::set_edge_strain_constraints(float wc)
	{
		Eigen::MatrixXi edges = get_edges();

		for (int i = 0; i < edges.rows(); i++)
		{
			const auto edge = edges.row(i);
			const auto e0 = edge(0);
			const auto e1 = edge(1);

			const double rest_length = (p.row(e0) - p.row(e1)).norm();
			constexpr double EPS = 1e-5;

			// discard very short edge
			if (rest_length < EPS)
			{
				continue;
			}

			constraints.push_back(new EdgeStrainConstraint(
				wc, e0, e1, p
			));
		}
	}

	void DeformableMesh::set_bending_constraints(float wc)
	{
		std::vector<bool> borders = igl::is_border_vertex(boundary_facets);
		for (int i = 0; i < positions().rows(); i++)
		{
			bool discard_flag = false;
			if (borders[i] == true)
			{
				// fixed_vertices.insert(i);
				discard_flag = true;
			}

			std::vector<int> neighbor_vertices;
			for (const int v : adj_list.at(i))
			{
				const double edge_length = (p.row(i) - p.row(v)).norm();
				constexpr double EPS = 1e-5;
				if (edge_length < EPS) // vertex that adj to a very short edge is not considered
				{
					discard_flag = true;
					break;
				}

				neighbor_vertices.push_back(v);
			}

			if (discard_flag)
			{
				printf("Discard bending in vertex %d\n", i);
				continue;
			}

			constraints.push_back(new BendingConstraint(
				wc, i, neighbor_vertices, p
			));
		}
	}

	void DeformableMesh::set_tet_strain_constraints(float wc)
	{

	}

	bool DeformableMesh::apply_mass_per_vertex(float mass_per_vertex)
	{
		bool need_modify = false;
		
		const auto eq = [](float a, float b)
		{
			constexpr float eps = 1e-5;
			float diff = std::abs(a - b);
			return diff <= eps;
		};

		for (int i = 0; i < p.rows(); i++)
		{
			if (eq(static_cast<float>(m(i)), mass_per_vertex) == false)
			{
				m(i) = static_cast<double>(mass_per_vertex);
				need_modify = true;
			}
		}
		
		return need_modify;
	}

	void DeformableMesh::resolve_collision(const std::unordered_map<int, std::unique_ptr<primitive::Primitive>>& rigid_colliders, Eigen::MatrixX3f& q_explicit) const
	{
		static int collision_cnt = 0;
		for (int i = 0; i < q_explicit.rows(); i++)
		{
			Eigen::Vector3f pos = q_explicit.row(i).transpose();
			for (const auto& [id, collider] : rigid_colliders)
			{
				bool flag = collider->collision_handle(pos);
				if (flag == true)
				{
					collision_cnt++;
					// printf("Detect collision! %d\n", collision_cnt);
				}
			}
			q_explicit.row(i) = pos.transpose();
		}
	}
}