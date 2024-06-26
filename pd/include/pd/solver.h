#pragma once

#include <array>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/SparseCore>

#include <pd/linear_sys_solver.h>
#include <pd/algo_ctrl.h>

#include <util/progress_percentage.h>
#include <util/cpu_timer.h>
#include <util/singleton.h>

namespace primitive
{
	class primitive;
}

namespace pd
{
	class GpuLocalSolver;

	class Solver: public util::Singleton<Solver>
	{
	public:
		Solver(token);
		~Solver();

		void free_cuda_memory();

		void set_dt(SimScalar dt)
		{
			this->dt = dt;
		}

		// Precompute A = LU(Cholesky) in vanilla PD
		// Precompute A products and A coefficients in A-Jacobi
		void precompute_A(const std::unordered_map<MeshIDType, DeformableMesh>& models);
		void precompute(const std::unordered_map<MeshIDType, DeformableMesh>& models);
		void step(
			std::unordered_map<MeshIDType, DeformableMesh>& models,
			const std::unordered_map<MeshIDType, DataMatrixX3>& f_exts,
			const std::unordered_map<pd::MeshIDType, std::unique_ptr<primitive::Primitive>>& rigid_colliders,
			int n_itr,
			int itr_solver_n_itr
		);
		void set_solver(pd::LinearSysSolver sel);
		void clear_solver();

		bool is_algo_changed() const { return algo_changed; }
		void set_algo_changed(bool val) { algo_changed = val; }

		bool is_dirty() const { return dirty; }
		void set_dirty(bool val);
		// timer variable
		util::CpuTimer timer;
		double last_local_step_time{ 0 };
		double last_global_step_time{ 0 };
		double last_precomputation_time{ 0 };

		// if use gpu for local step, we need to create virtual function table on the gpu
		bool use_gpu_for_local_step{ true };
		std::unique_ptr<GpuLocalSolver> gpu_local_solver{ nullptr };

		void set_chebyshev_params(SimScalar rho, SimScalar under_relaxation);
		void set_linear_sys_solver(int idx);
		static Float get_precompute_progress();

	private:
		// progress indicator
		static util::ProgressPercentage progress_percentage;

		// dirty = true indicates the solver needs to recompute
		bool dirty{ false };

		// algo_changed = true indicates setting new algorithm for the solver
		bool algo_changed{ false };

		// solver params
		SimScalar dt;

		// model
		Eigen::SparseMatrix<SimScalar> A;

		constexpr static int N_SOLVERS = 5;
		std::array<std::unique_ptr<LinearSystemSolver>, N_SOLVERS> solvers;
		std::array<std::unique_ptr<LinearSystemSolver>, N_SOLVERS>::iterator linear_sys_solver;

		// local step
		void local_step_cpu(const std::unordered_map<MeshIDType, DeformableMesh>& models, const SimPositions& q_nplus1, SimVectorX& b);
		void local_step_gpu(const SimPositions& q_nplus1, SimVectorX& b);
	};
}
