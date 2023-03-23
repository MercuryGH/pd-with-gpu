#include <pd/deformable_mesh.h>
#include <pd/positional_constraint.h>
#include <pd/edge_length_constraint.h>

namespace pd
{
	void DeformableMesh::toggle_vertices_fixed(const std::unordered_set<int> &v, float wi)
	{
		for (const int vi : v)
		{
			if (vertex_fixed[vi] == false)
			{
				// m(vi) = FIXED_VERTEX_MASS;
				add_positional_constraint(vi, wi);
			}
			vertex_fixed[vi] = !vertex_fixed[vi];
			// else
			// {
			// 	m(vi) = mass_per_vertex;
			// }
		}
	}

	void DeformableMesh::add_positional_constraint(int vi, float wi)
	{
		constraints.emplace_back(std::make_unique<PositionalConstraint>(
			wi, vi, p
		));
	}

	void DeformableMesh::set_edge_length_constraint(float wi)
	{
		Eigen::MatrixXi edges = get_edges();

		for (int i = 0; i < edges.rows(); i++)
		{
			const auto edge = edges.row(i);
			const auto e0 = edge(0);
			const auto e1 = edge(1);

			constraints.emplace_back(std::make_unique<EdgeLengthConstraint>(
				wi, e0, e1, p
			));
		}
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
			if (vertex_fixed[i] == true)
			{
				continue;
			}

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
					printf("Detect collision! %d\n", collision_cnt);
				}
			}
			q_explicit.row(i) = pos.transpose();
		}
	}
}