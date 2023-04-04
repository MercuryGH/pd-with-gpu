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
		/*
		DeformableMesh() :
			p0(p),
			p(p),
			f(f),
			e(e),
			m(m),
			v(p.rows(), p.cols()),
			fixed_vertices()
		{
			v.setZero(); // init velocity to 0
		}
		*/

		// construct from tetrahedron elements
		DeformableMesh(const Positions &p, const Elements &t, const Faces &boundary_facets, int obj_id) :
			p0(p),
			p(p),
			e(t),
			boundary_facets(boundary_facets),
			m(p.rows()),
			v(p.rows(), p.cols()),
			fixed_vertices(),
			obj_id(obj_id)
		{
			m.setOnes(); // Init messes to equally distributed
			v.setZero(); // init velocity to 0

			// Bug: block(5,3,4) etc. is a vertex non-manifold
			Eigen::VectorXi indicator;
			if (igl::is_vertex_manifold(boundary_facets, indicator) == false)
			{
				printf("Warning: Non vertex manifold mesh detected!\n");
			}
			if (igl::is_edge_manifold(boundary_facets) == false)
			{
				printf("Warning: Non edge manifold mesh detected!\n");
			}

			igl::adjacency_list(boundary_facets, adj_list, true);
		}

		// construct from triangle elements
		DeformableMesh(const Positions &p, const Elements &f, int obj_id) :
			p0(p),
			p(p),
			e(f),
			boundary_facets(f),
			m(p.rows()),
			v(p.rows(), p.cols()),
			fixed_vertices(),
			obj_id(obj_id)
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
			std::cout << "Debug: delete mesh " << this->obj_id << "\n";
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
		const Positions& positions() const { return p; }
		const Faces& faces() const { return boundary_facets; }
		const thrust::host_vector<pd::Constraint*>& get_all_constraints() const { return constraints; }
		bool is_vertex_fixed(int vi) const { return fixed_vertices.find(vi) != fixed_vertices.end(); };
		const std::unordered_set<int>& get_fixed_vertices() const { return fixed_vertices; }
		const std::vector<std::vector<int>>& get_adj_list() const { return adj_list; }
		Eigen::MatrixXi get_edges() const { Eigen::MatrixXi edges; igl::edges(e, edges); return edges; }
		const Eigen::VectorXd& get_masses() const { return m; }

		// const 

		// setters
		void reset_constraints()
		{
			v.setZero();
			constraints.clear();
			fixed_vertices.clear();
		}

		void update_positions_and_velocities(const Positions& p, const Velocities& v)
		{
			this->p = p;
			this->v = v;
		}

		void set_positions(const Positions& p)
		{
			this->p = p;
		}

		int obj_id{ -1 };

	    // methods
		void toggle_vertices_fixed(const std::unordered_set<int>& v, float wc);
		void set_edge_strain_constraints(float wc);
		void set_bending_constraints(float wc);
		void set_tet_strain_constraints(float wc);

		// TODO: use area weighted method to apply mass
		bool apply_mass_per_vertex(float mass_per_vertex);
		int n_edges{ 0 };   // #Edges

		void resolve_collision(const std::unordered_map<int, std::unique_ptr<primitive::Primitive>>& rigid_colliders, Eigen::MatrixX3f& q_explicit) const;

	private:
		void add_positional_constraint(int vi, float wc);

		Positions p0;  // Rest positions
		Positions p;   // Positions
		Faces boundary_facets; // for rendering only

		// Indicates the model is of 
		// triangle elements (=faces) or tetrahedra elements.
		// Dimensions may differ between different elements.
		// We need to restore the edges information from the elements matrix.
		Elements e; 

		Masses m;      // Per-vertex mass
		Velocities v;  // Per-vertex velocity
		thrust::host_vector<pd::Constraint*> constraints; // Vector of constraints

		std::vector<std::vector<int>> adj_list; // sorted adjancecy list

		std::unordered_set<int> fixed_vertices; // store all fixed vertex
	};
}