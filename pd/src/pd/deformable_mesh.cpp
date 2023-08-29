#include <pd/deformable_mesh.h>

#include <igl/adjacency_list.h>

#include <igl/is_vertex_manifold.h>
#include <igl/is_edge_manifold.h>

#include <igl/edges.h>
#include <igl/is_border_vertex.h>
#include <igl/adjacency_list.h>
#include <igl/barycenter.h>

#include <pd/positional_constraint.h>
#include <pd/edge_strain_constraint.h>
#include <pd/bending_constraint.h>
#include <pd/tet_strain_constraint.h>

#include <primitive/primitive.h>

#include <thrust/host_vector.h>
#include <thrust/universal_vector.h>

namespace pd
{
	struct DeformableMesh::pImpl final
	{
		pImpl(const PositionData &p, const ElementData &t, const FaceData &boundary_facets):
			p(p),
			e(t),
			boundary_facets(boundary_facets),
			m(p.rows()),
			vertex_normals(p.rows(), p.cols()),
			v(p.rows(), p.cols()),
			tet_mesh(true)
		{
			m.setOnes();
			v.setZero();
		}
		pImpl(const PositionData &p, const ElementData &f):
			p(p),
			e(f),
			boundary_facets(f),
			m(p.rows()),
			vertex_normals(p.rows(), p.cols()),
			v(p.rows(), p.cols()),
			tet_mesh(false)
		{
			m.setOnes();
			v.setZero();
		}
		~pImpl()
		{
			for (const auto& constraint : constraints)
			{
				delete constraint;
			}
			constraints.clear();
		}

		PositionData p;   // Positions
		PositionData barycenters; // barycenter positions (for tetrahedron visualization only)
		DataMatrixX3 vertex_normals; // vertex normals
		FaceData boundary_facets; // for rendering only

		// Indicates the model is of
		// triangle elements (=faces) or tetrahedra elements.
		// Dimensions may differ between different elements.
		// We need to restore the edges information from the elements matrix.
		ElementData e;
		bool tet_mesh{ false };

		MassData m;      // Per-vertex mass
		VelocityData v;  // Per-vertex velocity
		thrust::host_vector<pd::Constraint*> constraints; // Vector of constraints

		std::vector<std::vector<VertexIndexType>> adj_list; // adjancecy list (tri mesh: sorted, tet mesh: not sorted)

		std::unordered_set<VertexIndexType> fixed_vertices; // store all fixed vertex
	};

	// construct from tetrahedron elements
	DeformableMesh::DeformableMesh(const PositionData &p, const ElementData &t, const FaceData &boundary_facets)
	{
		p_impl = std::make_unique<pImpl>(p, t, boundary_facets);

		// Eigen::VectorXi indicator;
		// if (igl::is_vertex_manifold(boundary_facets, indicator) == false)
		// {
		// 	printf("Warning: Non vertex manifold mesh detected!\n");
		// }
		// if (igl::is_edge_manifold(boundary_facets) == false)
		// {
		// 	printf("Warning: Non edge manifold mesh detected!\n");
		// }

		// do not construct ordered adj list since the boundary of tet mesh may be non-manifold
		igl::adjacency_list(p_impl->boundary_facets, p_impl->adj_list, false);
		igl::barycenter(p_impl->p, p_impl->e, p_impl->barycenters);
	}

	// construct from triangle elements
	DeformableMesh::DeformableMesh(const PositionData &p, const ElementData &f)
	{
		p_impl = std::make_unique<pImpl>(p, f);

		Eigen::VectorXi indicator;
		if (igl::is_vertex_manifold(f, indicator) == false)
		{
			printf("Warning: Non vertex manifold mesh detected!\n");
		}
		if (igl::is_edge_manifold(f) == false)
		{
			printf("Warning: Non edge manifold mesh detected!\n");
		}

		igl::adjacency_list(f, p_impl->adj_list, true);
		if (p_impl->adj_list.size() != p_impl->p.rows())
		{
			printf("Warning: Singular vertex that doesn't have any neighbors detected!\n");
		}
	}

	DeformableMesh::DeformableMesh(const DeformableMesh& rhs)
	{
		// deep copy
		// std::cout << "DeformableMesh copy ctr\n";
		p_impl.reset(new pImpl(*rhs.p_impl));
	}
	DeformableMesh::DeformableMesh(DeformableMesh&& rhs) noexcept
	{
		// std::cout << "DeformableMesh move ctr\n";

		p_impl = std::move(rhs.p_impl);

		// std::cout << p_impl->p.row(0) << "\n";
	}
	DeformableMesh& DeformableMesh::operator=(const DeformableMesh& rhs)
	{
		// std::cout << "DeformableMesh copy assign\n";

		// deep copy
        if (this != &rhs)
        {
			p_impl.reset(new pImpl(*rhs.p_impl));
        }
        return *this;
	}
	DeformableMesh& DeformableMesh::operator=(DeformableMesh&& rhs) noexcept
	{
		// std::cout << "DeformableMesh move assign\n";

        if (this != &rhs)
        {
			p_impl = std::move(rhs.p_impl);
        }
        return *this;
	}

	DeformableMesh::~DeformableMesh()
	{
		// std::cout << "DeformableMesh destrctor " << "\n";
	}

	bool DeformableMesh::empty() const { return p_impl->p.rows() == 0; }
	const PositionData& DeformableMesh::positions() const
	{
		return p_impl->p;
	}
	const VelocityData& DeformableMesh::velocities() const { return p_impl->v; }
	const FaceData& DeformableMesh::faces() const { return p_impl->boundary_facets; }
	const thrust::host_vector<Constraint*>& DeformableMesh::get_all_constraints() const { return p_impl->constraints; }
	bool DeformableMesh::is_vertex_fixed(VertexIndexType vi) const { return p_impl->fixed_vertices.find(vi) != p_impl->fixed_vertices.end(); };
	const std::unordered_set<int>& DeformableMesh::get_fixed_vertices() const { return p_impl->fixed_vertices; }
	const std::vector<std::vector<VertexIndexType>>& DeformableMesh::get_adj_list() const { return p_impl->adj_list; }
	int DeformableMesh::n_constraints() const { return p_impl->constraints.size(); }
	bool DeformableMesh::is_tet_mesh() const { return p_impl->tet_mesh; }
	const MassData& DeformableMesh::masses() const { return p_impl->m; }
	const PositionData& DeformableMesh::get_element_barycenters() const { return p_impl->barycenters; }
	const ElementData& DeformableMesh::get_elements() const { return p_impl->e; }
	DataMatrixX3& DeformableMesh::vertex_normals() const { return p_impl->vertex_normals; }
	// setters
	void DeformableMesh::reset_constraints()
	{
		p_impl->v.setZero();
		p_impl->constraints.clear();
		p_impl->fixed_vertices.clear();
	}

	void DeformableMesh::apply_translation(DataVector3 translate)
	{
		for (int i = 0; i < p_impl->p.rows(); i++)
		{
			p_impl->p.row(i) += translate.transpose();
		}
	}
	void DeformableMesh::set_positions(const PositionData& p)
	{
		p_impl->p = p;
	}

	void DeformableMesh::update_positions_and_velocities(const PositionData& p, const VelocityData& v)
	{
		p_impl->p = p;
		p_impl->v = v;
	}
	void DeformableMesh::set_vertex_mass(VertexIndexType vid, DataScalar mass) { p_impl->m(vid) = mass; }

	Eigen::MatrixX2i DeformableMesh::get_edges() const
	{
		Eigen::MatrixX2i edges;
		igl::edges(p_impl->e, edges);
		return edges;
	}

	void DeformableMesh::toggle_vertices_fixed(const std::unordered_set<VertexIndexType> &v, SimScalar wc)
	{
		for (const VertexIndexType vi : v)
		{
			if (is_vertex_fixed(vi) == false)
			{
				add_positional_constraint(vi, wc);
				p_impl->fixed_vertices.insert(vi);
			}
			else
			{
				p_impl->fixed_vertices.erase(vi);
			}
		}
	}

	void DeformableMesh::add_positional_constraint(VertexIndexType vi, SimScalar wc)
	{
		p_impl->constraints.push_back(new PositionalConstraint(
			wc, vi, p_impl->p
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

			const DataScalar rest_length = (p_impl->p.row(e0) - p_impl->p.row(e1)).norm();
			constexpr DataScalar EPS = 1e-5;

			// discard very short edge
			if (rest_length < EPS)
			{
				continue;
			}

			p_impl->constraints.push_back(new EdgeStrainConstraint(
				wc, e0, e1, p_impl->p
			));
		}
	}

	void DeformableMesh::set_bending_constraints(SimScalar wc, bool discard_quadratic_term)
	{
		std::vector<bool> borders = igl::is_border_vertex(p_impl->boundary_facets);
		for (int i = 0; i < positions().rows(); i++)
		{
			bool discard_flag = false;
			if (borders[i] == true)
			{
				// fixed_vertices.insert(i);
				discard_flag = true;
			}

			std::vector<VertexIndexType> neighbor_vertices;
			for (const VertexIndexType v : p_impl->adj_list.at(i))
			{
				const DataScalar edge_length = (p_impl->p.row(i) - p_impl->p.row(v)).norm();
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

			p_impl->constraints.push_back(new BendingConstraint(
				wc, i, neighbor_vertices, p_impl->p, discard_quadratic_term
			));
		}
	}

	void DeformableMesh::set_tet_strain_constraints(SimScalar wc, SimVector3 min_strain_xyz, SimVector3 max_strain_xyz)
	{
		if (is_tet_mesh() == false)
		{
			return;
		}

		for (int i = 0; i < p_impl->e.rows(); i++)
		{
			const IndexRowVector4 tet_vertices = p_impl->e.row(i);

			p_impl->constraints.push_back(new TetStrainConstraint(
				wc, p_impl->p, tet_vertices, min_strain_xyz, max_strain_xyz
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

		for (int i = 0; i < p_impl->p.rows(); i++)
		{
			if (eq(static_cast<DataScalar>(p_impl->m(i)), mass_per_vertex) == false)
			{
				p_impl->m(i) = static_cast<DataScalar>(mass_per_vertex);
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
