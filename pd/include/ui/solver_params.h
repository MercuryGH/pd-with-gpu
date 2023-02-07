#pragma once

namespace ui
{
	enum class LinearSysSolver
	{
		DIRECT = 0,
		PARALLEL_JACOBI = 1,
		A_JACOBI = 2
	};
	struct SolverParams
	{
		float dt{ 0.0166667f };
		int n_solver_iterations{ 10 };

		LinearSysSolver selected_solver{ LinearSysSolver::DIRECT };
	};
}