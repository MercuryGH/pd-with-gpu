#pragma once

#include <array>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <pd/deformable_mesh.h>

#include <primitive/primitive.h>

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
		Solver() = delete;
		Solver(
			std::unordered_map<int, DeformableMesh>& models, 
			std::unordered_map<int, std::unique_ptr<primitive::Primitive>>& rigid_colliders
		);

		void set_dt(float dt)
		{
			this->dt = dt;
		}

		// Precompute A = LU(Cholesky) in vanilla PD
		// Precompute A products and A coefficients in A-Jacobi
		void precompute_A();
		void precompute();
		void step(const std::unordered_map<int, Eigen::MatrixX3d>& f_exts, int n_itr, int itr_solver_n_itr);
		void set_solver(ui::LinearSysSolver sel);
		void clear_solver();

		// algo_changed = true indicates setting new algorithm for the solver
		bool algo_changed{ false };

		// dirty = true indicates the solver needs to recompute
		bool dirty{ false };

		// timer variable
		util::CpuTimer timer;
		double last_local_step_time{ 0 };
		double last_global_step_time{ 0 };
		double last_precomputation_time{ 0 };

		// if use gpu for local step, we need to create virtual function table on the gpu
		bool use_gpu_for_local_step{ true };
		std::unique_ptr<GpuLocalSolver> gpu_local_solver{ nullptr };

	private:
		// solver params
		float dt;

		// model
		std::unordered_map<int, DeformableMesh>& models;
		std::unordered_map<int, std::unique_ptr<primitive::Primitive>>& rigid_colliders;
		Eigen::SparseMatrix<float> A;

		constexpr static int N_SOLVERS = 5;
		std::array<std::unique_ptr<LinearSystemSolver>, N_SOLVERS> solvers;
		std::array<std::unique_ptr<LinearSystemSolver>, N_SOLVERS>::iterator linear_sys_solver;

		// local step
		void local_step_cpu(const Eigen::VectorXf& q_nplus1, Eigen::VectorXf& b);
		void local_step_gpu(const Eigen::VectorXf& q_nplus1, Eigen::VectorXf& b);
	};
}
