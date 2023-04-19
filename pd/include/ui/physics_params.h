#pragma once

#include <Eigen/Core>

namespace ui
{
	struct PhysicsParams
	{
		bool enable_gravity{ true };
		float mass_per_vertex{ 0.001f };

		float edge_strain_constraint_wc{ 100.f }; 
		float positional_constraint_wc{ 100.f };
		float bending_constraint_wc{ 5e-7f };
		float tet_strain_constraint_wc{ 0.1f };
		Eigen::Vector3f tet_strain_constraint_min_xyz{ 0.95, 0.95, 0.95 };
		Eigen::Vector3f tet_strain_constraint_max_xyz{ Eigen::Vector3f::Ones() };

		float external_force_val{ 0.1f };
	};
}