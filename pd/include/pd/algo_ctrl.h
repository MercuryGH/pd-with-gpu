#pragma once

#include <unordered_set>

#include <Eigen/Core>

namespace pd {
	struct PhysicsParams final
	{
		bool enable_gravity{ true };
		double mass_per_vertex{ 0.001 };

		float edge_strain_constraint_wc{ 100.f };
		float positional_constraint_wc{ 100.f };
		float bending_constraint_wc{ 5e-7f };
		bool discard_bending_constraint_quadratic_term_when_setting{ false };

		float tet_strain_constraint_wc{ 1000.f };
		Eigen::Vector3f tet_strain_constraint_min_xyz{ 0.95, 0.95, 0.95 };
		Eigen::Vector3f tet_strain_constraint_max_xyz{ Eigen::Vector3f::Ones() };

		float external_force_val{ 0.1f };
		Eigen::Vector3f external_force_dir_cache{ 0, 0, 1 }; // for API invoke

		bool enable_wind{ false };
		float wind_force_val{ 0.01f };
		Eigen::Vector3f wind_dir{ 0, 0, 1 };
	};

	enum class LinearSysSolver
	{
		DIRECT = 0,
		PARALLEL_JACOBI = 1,
		A_JACOBI_1 = 2,
		A_JACOBI_2 = 3,
		A_JACOBI_3 = 4,
	};

	struct SolverParams final
	{
		double dt{ 0.0166667 }; // use double precision to avoid floating point numeric error
		int n_solver_pd_iterations{ 10 };
		int n_itr_solver_iterations{ 500 };

		bool use_gpu_for_local_step{ true };

		// params for chebyshev method
		double rho{ 0.9992 };
		double under_relaxation{ 1 };

		LinearSysSolver selected_solver{ LinearSysSolver::DIRECT };
	};

	struct UserControl final
	{
		// Depends on user controled mesh_id
		int apply_ext_force_mesh_id{ 0 };
		bool apply_ext_force{ false };
		int ext_forced_vertex_idx{ 0 };
		int mouse_x{ 0 };
		int mouse_y{ 0 };

		// User LMC a vertex
		int selected_vertex_idx{ 0 };

		// This does not depend on mesh_id. It depends on the id of the current selected mesh.
		int cur_sel_mesh_id{ -1 };
		bool toggle_vertex_fix{ false };
		std::unordered_set<int> toggle_fixed_vertex_idxs;
		std::unordered_set<int> vertex_idxs_memory;

		bool always_recompute_normal{ false }; // rendering but not very costly

		// debug draw (adj vertex)
		bool enable_debug_draw{ false };
		bool enable_tetrahedra_visualization{ false };

		// headless mode: disable rendering, set to true when act as an algo library
		bool headless_mode{ false };
	};
}
