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
#include <util/cpu_timer.h>
#include <pd/gpu_local_solver.h>

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
		void step(const Eigen::MatrixXd& f_ext, int n_itr, int itr_solver_n_itr);
		void set_solver(ui::LinearSysSolver sel);
		void clear_solver();

		// algo_changed = true indicates setting new algorithm for the solver
		bool algo_changed{ false };

		// dirty = true indicates the solver needs to recompute
		bool dirty{ false };

		// timer variable
		util::CpuTimer timer;
		double last_local_step_time;
		double last_global_step_time;

		// if use gpu for local step, we need to create virtual function table on the gpu
		bool use_gpu_for_local_step{ false };
		std::unique_ptr<GpuLocalSolver> gpu_local_solver{ nullptr };

	private:
		// solver params
		float dt;

		// model
		DeformableMesh* model;
		Eigen::SparseMatrix<float> A;

		LinearSystemSolver* linear_sys_solver;
		constexpr static int N_SOLVERS = 5;
		std::array<LinearSystemSolver*, N_SOLVERS> solvers;

		// local step
		void local_step_cpu(const Eigen::VectorXf& q_nplus1, Eigen::VectorXf& b);
		void local_step_gpu(const Eigen::VectorXf& q_nplus1, Eigen::VectorXf& b);
	};
}
