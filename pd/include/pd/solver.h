#pragma once

#include <array>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <pd/deformable_mesh.h>

#include <pd/linear_sys_solver.h>
#include <pd/cholesky_direct.h>
#include <pd/parallel_jacobi.h>
#include <pd/a_jacobi.h>

#include <ui/solver_params.h>

namespace pd
{
	class Solver
	{
	public:
		Solver();

		void set_model(DeformableMesh* model)
		{
			this->model = model;
			this->dirty = true;
		}

		void set_dt(float dt)
		{
			this->dt = dt;
		}

		// Precompute A = LU(Cholesky) in vanilla PD
		// Precompute A products and A coefficients in A-Jacobi
		void precompute_A();
		void precompute();
		void step(const Eigen::MatrixXd& f_ext, int n_iterations);
		void set_solver(ui::LinearSysSolver sel);

		// dirty = true indicates the solver needs to recompute
		bool dirty{ false };

	private:
		// solver params
		float dt;

		// model
		DeformableMesh* model;
		Eigen::SparseMatrix<float> A;

		LinearSystemSolver* linear_sys_solver;
		constexpr static int N_SOLVERS = 3;
		std::array<LinearSystemSolver*, N_SOLVERS> solvers;
	};
}
