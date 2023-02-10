#pragma once

#include <iostream>
#include <vector>
#include <unordered_set>
#include <Eigen/Core>
#include <numeric>
#include <pd/constraint.h>

#include <pd/types.h>

namespace pd
{
	class DeformableMesh {
		friend class Solver;
		friend class AJacobi;
	public:
		DeformableMesh() = default;
		/*
		DeformableMesh(Positions p, Faces f, Elements e, Masses m) :
			p0(p),
			p(p),
			f(f),
			e(e),
			m(m),
			v(p.rows(), p.cols()),
			vertex_fixed(p.rows(), false)
		{
			v.setZero(); // init velocity to 0
		}
		*/
		DeformableMesh(Positions p, Faces f, Elements e) :
			p0(p),
			p(p),
			f(f),
			e(e),
			m(p.rows()),
			v(p.rows(), p.cols()),
			vertex_fixed(p.rows(), false)
		{
			m.setOnes(); // Init messes to equally distributed
			v.setZero(); // init velocity to 0
		}

		// Debug only
		void dimension_check() const 
		{
			assert(m.rows() == p.rows());
		}

		// getters
		bool empty() const { return p.rows() == 0; }
		const Positions& positions() const { return p; }
		const Faces& faces() const { return f; }
		size_t n_constraints() const { return constraints.size(); }
		bool is_vertex_fixed(int vi) const { return vertex_fixed[vi]; };

		// setters
		void reset_constraints()
		{
			v.setZero();
			constraints.clear();
			for (int i = 0; i < vertex_fixed.size(); i++)
				vertex_fixed[i] = false;
		}

		void update_positions_and_velocities(const Positions& p, const Velocities& v)
		{
			this->p = p;
			this->v = v;
		}

	    // methods
		void toggle_vertices_fixed(const std::unordered_set<int>& v, float wi, float mass_per_vertex);
		void set_edge_length_constraint(float wi);
		bool apply_mass_per_vertex(float mass_per_vertex);

	private:
		void add_positional_constraint(int vi, float wi, float mass_per_vertex);

		constexpr static float FIXED_VERTEX_MASS = 1e10; // fixed point has +inf mass and cannot move

		Positions p0;  // Rest positions
		Positions p;   // Positions
		Faces f;
		Elements e;    // Indicates the model is of 
		// triangle elements (=faces) or tetrahedra elements (currently not used in this proj)
		// Dimensions may differ between different elements.
		// Note: we need to restore the edges information from the elements matrix
		Masses m;      // Per-vertex mass
		Velocities v;
		Constraints constraints; // Vector of constraints
		std::vector<bool> vertex_fixed; // Indicates if a vertex is fixed
	};
}