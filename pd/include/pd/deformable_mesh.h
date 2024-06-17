#pragma once

#include <iostream>
#include <vector>
#include <numeric>
#include <unordered_set>
#include <memory>

#include <Eigen/Core>

#include <pd/types.h>

#include <primitive/primitive.h>



namespace pd
{
	class Constraint; // forward declaration to hide file constraint.h

	class DeformableMesh {
		friend class Solver;
		friend class AJacobi;

	private:
		struct pImpl;
		std::unique_ptr<pImpl> p_impl;

	public:
		DeformableMesh() = default;

		// construct from tetrahedron elements
		DeformableMesh(const PositionData &p, const ElementData &t, const FaceData &boundary_facets);
		DeformableMesh(const PositionData &p, const ElementData &f);

		// copy and move constructor
		DeformableMesh(const DeformableMesh& rhs);
		DeformableMesh(DeformableMesh&& rhs) noexcept;
		DeformableMesh& operator=(const DeformableMesh& rhs);
		DeformableMesh& operator=(DeformableMesh&& rhs) noexcept;

		~DeformableMesh();

		// getters
		bool empty() const;
		const PositionData& positions() const;
		const VelocityData& velocities() const;
		const FaceData& faces() const;
		// forward declaration needs a default for the second template parameter
		const std::vector<Constraint*, std::allocator<Constraint*>>& get_all_constraints() const;
		bool is_vertex_fixed(VertexIndexType vi) const;
		const std::unordered_set<int>& get_fixed_vertices() const;
		const std::vector<std::vector<VertexIndexType>>& get_adj_list() const;
		int n_constraints() const;
		bool is_tet_mesh() const;
		const PositionData& get_element_barycenters() const;
		const ElementData& get_elements() const;
		DataMatrixX3& vertex_normals() const;

		void set_vertex_mass(VertexIndexType vid, DataScalar mass);

		/**
		 * @brief Get the edges from elements
		 * @note edges are restored from triangles or tetrahedra data
		 * @return Eigen::MatrixXi #edges*2 array of integers
		 */
		Eigen::MatrixX2i get_edges() const;
		const MassData& masses() const;
		// setters
		void reset_constraints();
		void update_positions_and_velocities(const PositionData& p, const VelocityData& v);
		void set_positions(const PositionData& p);
		void apply_translation(DataVector3 translate);

	    // methods
		void toggle_vertices_fixed(const std::unordered_set<VertexIndexType>& v, SimScalar wc);
		void set_edge_strain_constraints(SimScalar wc);
		void set_bending_constraints(SimScalar wc, bool discard_quadratic_term=false);
		void set_tet_strain_constraints(SimScalar wc, SimVector3 min_strain_xyz=SimVector3::Ones(), SimVector3 max_strain_xyz=SimVector3::Ones());

		// Currently use uniform weighted mass. Alternative: use area weighted mass.
		bool apply_mass_per_vertex(DataScalar mass_per_vertex);
		int n_edges{ 0 };   // #Edges

		static void resolve_collision(const std::unordered_map<pd::MeshIDType, std::unique_ptr<primitive::Primitive>>& rigid_colliders, SimMatrixX3& q_explicit);

		void add_positional_constraint(VertexIndexType vi, SimScalar wc);
	};
}
