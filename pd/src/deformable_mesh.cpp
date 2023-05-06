#include <pd/deformable_mesh.h>
#include <pd/positional_constraint.h>
#include <pd/edge_strain_constraint.h>
#include <pd/bending_constraint.h>
#include <pd/tet_strain_constraint.h>

namespace pd
{
	void DeformableMesh::toggle_vertices_fixed(const std::unordered_set<VertexIndexType> &v, SimScalar wc)
	{
		for (const VertexIndexType vi : v)
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

	void DeformableMesh::add_positional_constraint(VertexIndexType vi, SimScalar wc)
	{
		constraints.push_back(new PositionalConstraint(
			wc, vi, p
		));
	}

	void DeformableMesh::set_edge_strain_constraints(SimScalar wc)
	{
		Eigen::MatrixX2i edges = get_edges();

		for (int i = 0; i < edges.rows(); i++)
		{
			const auto edge = edges.row(i);
			const auto e0 = edge(0);
			const auto e1 = edge(1);

			const DataScalar rest_length = (p.row(e0) - p.row(e1)).norm();
			constexpr DataScalar EPS = 1e-5;

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

	void DeformableMesh::set_bending_constraints(SimScalar wc, bool discard_quadratic_term)
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

			std::vector<VertexIndexType> neighbor_vertices;
			for (const VertexIndexType v : adj_list.at(i))
			{
				const DataScalar edge_length = (p.row(i) - p.row(v)).norm();
				constexpr DataScalar EPS = 1e-5;
				if (edge_length < EPS) // vertex that adj to a very short edge is not considered
				{
					discard_flag = true;
					break;
				}

				neighbor_vertices.push_back(v);
			}

			if (discard_flag)
			{
				// printf("Discard bending at vertex %d\n", i);
				continue;
			}

			constraints.push_back(new BendingConstraint(
				wc, i, neighbor_vertices, p, discard_quadratic_term
			));
		}
	}

	void DeformableMesh::set_tet_strain_constraints(SimScalar wc, SimVector3 min_strain_xyz, SimVector3 max_strain_xyz)
	{
		if (is_tet_mesh() == false)
		{
			return;
		}

		for (int i = 0; i < e.rows(); i++)
		{
			const IndexRowVector4 tet_vertices = e.row(i);

			constraints.push_back(new TetStrainConstraint(
				wc, p, tet_vertices, min_strain_xyz, max_strain_xyz
			));
		}
	}

	bool DeformableMesh::apply_mass_per_vertex(DataScalar mass_per_vertex)
	{
		bool need_modify = false;
		
		const auto eq = [](DataScalar a, DataScalar b)
		{
			constexpr DataScalar eps = 1e-5;
			DataScalar diff = std::abs(a - b);
			return diff <= eps;
		};

		for (int i = 0; i < p.rows(); i++)
		{
			if (eq(static_cast<DataScalar>(m(i)), mass_per_vertex) == false)
			{
				m(i) = static_cast<DataScalar>(mass_per_vertex);
				need_modify = true;
			}
		}
		
		return need_modify;
	}

	void DeformableMesh::resolve_collision(const std::unordered_map<pd::MeshIDType, std::unique_ptr<primitive::Primitive>>& rigid_colliders, SimMatrixX3& q_explicit)
	{
		static int collision_cnt = 0;
		for (int i = 0; i < q_explicit.rows(); i++)
		{
			SimVector3 pos = q_explicit.row(i).transpose();
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