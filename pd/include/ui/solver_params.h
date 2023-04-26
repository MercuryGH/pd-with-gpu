#pragma once

namespace ui
{
	enum class LinearSysSolver
	{
		DIRECT = 0,
		PARALLEL_JACOBI = 1,
		A_JACOBI_1 = 2,
		A_JACOBI_2 = 3,
		A_JACOBI_3 = 4,
	};

	struct SolverParams
	{
		double dt{ 0.0166667 }; // use double precision to avoid floating point numeric error
		int n_solver_pd_iterations{ 10 };
		int n_itr_solver_iterations{ 500 };

		bool use_gpu_for_local_step{ true };

		LinearSysSolver selected_solver{ LinearSysSolver::DIRECT };
	};
}