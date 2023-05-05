#pragma once

#include <iostream>
#include <vector>
#include <numeric>
#include <unordered_set>

#include <igl/adjacency_list.h>

#include <igl/is_vertex_manifold.h>
#include <igl/is_edge_manifold.h>

#include <igl/edges.h>
#include <igl/is_border_vertex.h>
#include <igl/adjacency_list.h>
#include <igl/barycenter.h>
#include <Eigen/Core>

#include <primitive/primitive.h>
#include <pd/constraint.h>
#include <pd/types.h>

#include <thrust/host_vector.h>
#include <thrust/universal_vector.h>

namespace pd
{
	class DeformableMesh {
		friend class Solver;
		friend class AJacobi;
	public:
		DeformableMesh() = default;

		// construct from tetrahedron elements
		DeformableMesh(const PositionData &p, const ElementData &t, const FaceData &boundary_facets, MeshIDType obj_id) :
			p0(p),
			p(p),
			e(t),
			boundary_facets(boundary_facets),
			m(p.rows()),
			v(p.rows(), p.cols()),
			fixed_vertices(),
			obj_id(obj_id),
			tet_mesh(true)
		{
			m.setOnes(); // Init messes to equally distributed
			v.setZero(); // init velocity to 0

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
			igl::adjacency_list(boundary_facets, adj_list, false);
			igl::barycenter(p, e, barycenters);
		}

		// construct from triangle elements
		DeformableMesh(const PositionData &p, const ElementData &f, MeshIDType obj_id) :
			p0(p),
			p(p),
			e(f),
			boundary_facets(f),
			m(p.rows()),
			v(p.rows(), p.cols()),
			fixed_vertices(),
			obj_id(obj_id),
			tet_mesh(false)
		{
			m.setOnes(); // Init messes to equally distributed
			v.setZero(); // init velocity to 0
			
			Eigen::VectorXi indicator;
			if (igl::is_vertex_manifold(f, indicator) == false)
			{
				printf("Warning: Non vertex manifold mesh detected!\n");
			}
			if (igl::is_edge_manifold(f) == false)
			{
				printf("Warning: Non edge manifold mesh detected!\n");
			}

			igl::adjacency_list(f, adj_list, true);
		}

		~DeformableMesh()
		{
			// std::cout << "Debug: delete mesh " << this->obj_id << "\n";
			for (const auto& constraint : constraints)
			{
				delete constraint;
			}
			constraints.clear();
		}

		// Debug only
		void dimension_check() const 
		{
			assert(m.rows() == p.rows());
		}

		// getters
		bool empty() const { return p.rows() == 0; }
		const PositionData& positions() const { return p; }
		const FaceData& faces() const { return boundary_facets; }
		const thrust::host_vector<pd::Constraint*>& get_all_constraints() const { return constraints; }
		bool is_vertex_fixed(VertexIndexType vi) const { return fixed_vertices.find(vi) != fixed_vertices.end(); };
		const std::unordered_set<int>& get_fixed_vertices() const { return fixed_vertices; }
		const std::vector<std::vector<VertexIndexType>>& get_adj_list() const { return adj_list; }
		int n_constraints() const { return constraints.size(); }
		bool is_tet_mesh() const { return tet_mesh; }
		const PositionData& get_element_barycenters() const { return barycenters; }
		const ElementData& get_elements() const { return e; }

		void set_vertex_mass(VertexIndexType vid, DataScalar mass) { m(vid) = mass; }

		/**
		 * @brief Get the edges from elements
		 * @note edges are restored from triangles or tetrahedra data
		 * @return Eigen::MatrixXi #edges*2 array of integers
		 */
		Eigen::MatrixX2i get_edges() const 
		{ 
			Eigen::MatrixX2i edges; 
			igl::edges(e, edges); 
			return edges; 
		}
		const MassData& get_masses() const { return m; }

		// setters
		void reset_constraints()
		{
			v.setZero();
			constraints.clear();
			fixed_vertices.clear();
		}

		void update_positions_and_velocities(const PositionData& p, const VelocityData& v)
		{
			this->p = p;
			this->v = v;
		}

		void set_positions(const PositionData& p)
		{
			this->p = p;
		}

		void apply_translation(DataVector3 translate)
		{
			for (int i = 0; i < p.rows(); i++)
			{
				p.row(i) += translate.transpose();
			}
		}

		MeshIDType obj_id{ -1 };

	    // methods
		void toggle_vertices_fixed(const std::unordered_set<VertexIndexType>& v, SimScalar wc);
		void set_edge_strain_constraints(SimScalar wc);
		void set_bending_constraints(SimScalar wc);
		void set_tet_strain_constraints(SimScalar wc, SimVector3 min_strain_xyz=SimVector3::Ones(), SimVector3 max_strain_xyz=SimVector3::Ones());

		// Currently use uniform weighted mass. Alternative: use area weighted mass.
		bool apply_mass_per_vertex(DataScalar mass_per_vertex);
		int n_edges{ 0 };   // #Edges

		static void resolve_collision(const std::unordered_map<pd::MeshIDType, std::unique_ptr<primitive::Primitive>>& rigid_colliders, SimMatrixX3& q_explicit);

	private:
		void add_positional_constraint(VertexIndexType vi, SimScalar wc);

		PositionData p0;  // Rest positions
		PositionData p;   // Positions
		PositionData barycenters; // barycenter positions (for tetrahedron visualization only)
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
}